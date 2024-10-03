
// Programme original realise en Fortran 77 en mars 1994
// pour les Travaux Pratiques de Modelisation Numerique
// DEA d'astrophysique et techniques spatiales de Meudon

// 	par Herve Aussel et Emmanuel Quemener

// Conversion en C par Emmanuel Quemener en aout 1997
// Modification par Emmanuel Quemener en aout 2019

// Licence CC BY-NC-SA Emmanuel QUEMENER <emmanuel.quemener@gmail.com>

// Remerciements a :
 
// - Herve Aussel pour sa procedure sur le spectre de corps noir
// - Didier Pelat pour l'aide lors de ce travail
// - Jean-Pierre Luminet pour son article de 1979
// - Le Numerical Recipies pour ses recettes de calcul
// - Luc Blanchet pour sa disponibilite lors de mes interrogations en RG

// Compilation sous clang
// export DPCPP_HOME=$PWD/sycl_workspace
// export PATH=$DPCPP_HOME/llvm/build/bin:$PATH
// export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH
// 
// clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -DFP32 trou_noir_SyCL.cpp -o trou_noir_SyCL_FP32 -lm
// ./trou_noir_SyCL_FP32

#include <iostream>
#include <sycl/sycl.hpp>
#include <sys/time.h>
#include <time.h>

#define nbr 256 /* Nombre de colonnes du spectre */

#define PI 3.14159265359

#define TRACKPOINTS 2048

#if TYPE == FP64
#define MYFLOAT double
#else
#define MYFLOAT float
#endif

using namespace sycl;
MYFLOAT atanp(MYFLOAT x,MYFLOAT y)
{
  MYFLOAT angle;

  angle=atan2(y,x);

  if (angle<0)
    {
      angle+=2*PI;
    }

  return angle;
}

using namespace sycl;
MYFLOAT f(MYFLOAT v)
{
  return v;
}

using namespace sycl;
MYFLOAT g(MYFLOAT u,MYFLOAT m,MYFLOAT b)
{
  return (3.*m/b*pow(u,2)-u);
}

using namespace sycl;
void calcul(MYFLOAT *us,MYFLOAT *vs,MYFLOAT up,MYFLOAT vp,
	    MYFLOAT h,MYFLOAT m,MYFLOAT b)
{
  MYFLOAT c[4],d[4];

  c[0]=h*f(vp);
  c[1]=h*f(vp+c[0]/2.);
  c[2]=h*f(vp+c[1]/2.);
  c[3]=h*f(vp+c[2]);
  d[0]=h*g(up,m,b);
  d[1]=h*g(up+d[0]/2.,m,b);
  d[2]=h*g(up+d[1]/2.,m,b);
  d[3]=h*g(up+d[2],m,b);

  *us=up+(c[0]+2.*c[1]+2.*c[2]+c[3])/6.;
  *vs=vp+(d[0]+2.*d[1]+2.*d[2]+d[3])/6.;
}

using namespace sycl;
void rungekutta(MYFLOAT *ps,MYFLOAT *us,MYFLOAT *vs,
		MYFLOAT pp,MYFLOAT up,MYFLOAT vp,
		MYFLOAT h,MYFLOAT m,MYFLOAT b)
{
  calcul(us,vs,up,vp,h,m,b);
  *ps=pp+h;
}

using namespace sycl;
MYFLOAT decalage_spectral(MYFLOAT r,MYFLOAT b,MYFLOAT phi,
			 MYFLOAT tho,MYFLOAT m)
{
  return (sqrt(1-3*m/r)/(1+sqrt(m/pow(r,3))*b*sin(tho)*sin(phi)));
}

using namespace sycl;
MYFLOAT spectre(MYFLOAT rf,MYFLOAT q,MYFLOAT b,MYFLOAT db,
	     MYFLOAT h,MYFLOAT r,MYFLOAT m,MYFLOAT bss)
{
  MYFLOAT flx;

  flx=exp(q*log(r/m))*pow(rf,4)*b*db*h;
  return(flx);
}

using namespace sycl;
MYFLOAT spectre_cn(MYFLOAT rf,MYFLOAT b,MYFLOAT db,
		MYFLOAT h,MYFLOAT r,MYFLOAT m,MYFLOAT bss)
{
  
  MYFLOAT flx;
  MYFLOAT nu_rec,nu_em,qu,temp_em,flux_int;
  int fi,posfreq;

#define planck 6.62e-34
#define k 1.38e-23
#define c2 9.e16
#define temp 3.e7
#define m_point 1.

#define lplanck (log(6.62)-34.*log(10.))
#define lk (log(1.38)-23.*log(10.))
#define lc2 (log(9.)+16.*log(10.))
  
  MYFLOAT v=1.-3./r;

  qu=1./sqrt((1.-3./r)*r)*(sqrt(r)-sqrt(6.)+sqrt(3.)/2.*log((sqrt(r)+sqrt(3.))/(sqrt(r)-sqrt(3.))* 0.17157287525380988 ));

  temp_em=temp*sqrt(m)*exp(0.25*log(m_point)-0.75*log(r)-0.125*log(v)+0.25*log(fabs(qu)));

  flux_int=0.;
  flx=0.;

  for (fi=0;fi<nbr;fi++)
    {
      nu_em=bss*(MYFLOAT)fi/(MYFLOAT)nbr;
      nu_rec=nu_em*rf;
      posfreq=(int)(nu_rec*(MYFLOAT)nbr/bss);
      if ((posfreq>0)&&(posfreq<nbr))
  	{
	  flux_int=2.*planck/c2*pow(nu_em,3)/(exp(planck*nu_em/(k*temp_em))-1.)*exp(3.*log(rf))*b*db*h;
  	  flx+=flux_int;
  	}
    }

  return((MYFLOAT)flx);
}

using namespace sycl;
void impact(MYFLOAT d,MYFLOAT phi,int dim,MYFLOAT r,MYFLOAT b,MYFLOAT tho,MYFLOAT m,
	    MYFLOAT *zp,MYFLOAT *fp,
	    MYFLOAT q,MYFLOAT db,
	    MYFLOAT h,MYFLOAT bss,int raie)
{
  MYFLOAT xe,ye;
  int xi,yi;
  MYFLOAT flx,rf;
  xe=d*sin(phi);
  ye=-d*cos(phi);

  xi=(int)xe+dim/2;
  yi=(int)ye+dim/2;
  
  rf=decalage_spectral(r,b,phi,tho,m);

  if (raie==0)
    {
      bss=1.e19;
      flx=spectre_cn(rf,b,db,h,r,m,bss);
    }
  else
    {
      bss=2.;
      flx=spectre(rf,q,b,db,h,r,m,bss);
    }
  
  if (zp[xi+dim*yi]==0.)
    {
      zp[xi+dim*yi]=1./rf;
    }
  
  if (fp[xi+dim*yi]==0.)
    {
      fp[xi+dim*yi]=flx;
    }

}

// void sauvegarde_pgm(*char nom[24],unsigned int *image,int dim)
void sauvegarde_pgm(char *nom,unsigned int *image,int dim)
{
  FILE            *sortie;
  unsigned long   i,j;
  
  sortie=fopen(nom,"w");
  
  fprintf(sortie,"P5\n");
  fprintf(sortie,"%i %i\n",dim,dim);
  fprintf(sortie,"255\n");

  for (j=0;j<dim;j++) for (i=0;i<dim;i++)
    {
      fputc(image[i+dim*j],sortie);
    }

  fclose(sortie);
}

using namespace std;
int main(int argc,char *argv[])
{

  MYFLOAT m,rs,ri,re,tho;
  MYFLOAT q;

  MYFLOAT bss,stp;
  int nmx,dim;
  MYFLOAT bmx;
  MYFLOAT *zp,*fp;
  unsigned int *izp,*ifp;
  MYFLOAT zmx,fmx;
  int zimx=0,zjmx=0,fimx=0,fjmx=0;
  int raie,fcl,zcl;
  struct timeval tv1,tv2;
  MYFLOAT elapsed,cputime,epoch;
  int mtv1,mtv2;
  unsigned int epoch1,epoch2;

  if (argc==2)
    {
      if (strcmp(argv[1],"-aide")==0)
        {
          printf("\nSimulation d'un disque d'accretion autour d'un trou noir\n");
          printf("\nParametres a definir :\n\n");
          printf("  1) Dimension de l'Image\n");
          printf("  2) Masse relative du trou noir\n");
          printf("  3) Dimension du disque exterieur\n");
          printf("  4) Inclinaison par rapport au disque (en degres)\n");
          printf("  5) Rayonnement de disque MONOCHROMATIQUE ou CORPS_NOIR\n");
          printf("  6) Impression des images NEGATIVE ou POSITIVE\n"); 
          printf("  7) Nom de l'image des Flux\n");
          printf("  8) Nom de l'image des decalages spectraux\n");
          printf("\nSi aucun parametre defini, parametres par defaut :\n\n");
          printf("  1) Dimension de l'image : 1024 pixels de cote\n");
          printf("  2) Masse relative du trou noir : 1\n");
          printf("  3) Dimension du disque exterieur : 12 \n");
          printf("  4) Inclinaison par rapport au disque (en degres) : 10\n");
          printf("  5) Rayonnement de disque CORPS_NOIR\n");
          printf("  6) Impression des images NEGATIVE ou POSITIVE\n"); 
          printf("  7) Nom de l'image des flux : flux.pgm\n");
          printf("  8) Nom de l'image des z : z.pgm\n");
        }
    }
  
  if ((argc==9)||(argc==7))
    {
      printf("# Utilisation les valeurs definies par l'utilisateur\n");
      
      dim=atoi(argv[1]);
      m=atof(argv[2]);
      re=atof(argv[3]);
      tho=PI/180.*(90-atof(argv[4]));
      
      rs=2.*m;
      ri=3.*rs;

      if (strcmp(argv[5],"CORPS_NOIR")==0)
	{
	  raie=0;
	}
      else
	{
	  raie=1;
	}

    }
  else
    {
      printf("# Utilisation les valeurs par defaut\n");
      
      dim=1024;
      m=1.;
      rs=2.*m;
      ri=3.*rs;
      re=12.;
      tho=PI/180.*80;
      // Corps noir
      raie=0;
    }

  if (raie==1)
    {
      bss=2.;
      q=-2;
    }
  else
    {
      bss=1.e19;
      q=-0.75;
    }

  printf("# Dimension de l'image : %i\n",dim);
  printf("# Masse : %f\n",m);
  printf("# Rayon singularite : %f\n",rs);
  printf("# Rayon interne : %f\n",ri);
  printf("# Rayon externe : %f\n",re);
  printf("# Inclinaison a la normale en radian : %f\n",tho);
  
  zp=(MYFLOAT*)calloc(dim*dim,sizeof(MYFLOAT));
  fp=(MYFLOAT*)calloc(dim*dim,sizeof(MYFLOAT));

  izp=(unsigned int*)calloc(dim*dim,sizeof(unsigned int));  
  ifp=(unsigned int*)calloc(dim*dim,sizeof(unsigned int));

  nmx=dim;
  stp=dim/(2.*nmx);
  bmx=1.25*re;

  // Set start timer
  gettimeofday(&tv1, NULL);
  mtv1=clock()*1000/CLOCKS_PER_SEC;
  epoch1=time(NULL);

  sycl::buffer<MYFLOAT> IZP(&zp[0],dim*dim);
  sycl::buffer<MYFLOAT> IFP(&fp[0],dim*dim);

  // Creating SYCL queue
  sycl::queue Queue;

  Queue.submit([&](auto &h) {
    sycl::accessor AZP{IZP, h};
    sycl::accessor AFP{IFP, h};
    
    // Executing kernel
    h.parallel_for(dim,[=](auto n) {
 
      MYFLOAT d,db,b,nh,h,up,vp,pp,us,vs,ps;
      MYFLOAT phi,phd,php,nr,r;
      int ni,ii,imx,tst;
      MYFLOAT rp[TRACKPOINTS];

      h=4.*PI/(MYFLOAT)TRACKPOINTS;
      d=stp*(MYFLOAT)(n+1);

      db=bmx/(MYFLOAT)nmx;
      b=db*(MYFLOAT)(n+1);
      up=0.;
      vp=1.;
      
      pp=0.;
      nh=1;
      
      rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);
      
      rp[(int)nh]=fabs(b/us);
      
      do
        {
          nh++;
          pp=ps;
          up=us;
          vp=vs;
          rungekutta(&ps,&us,&vs,pp,up,vp,h,m,b);
          
          rp[(int)nh]=b/us;
          
        } while ((rp[(int)nh]>=rs)&&(rp[(int)nh]<=rp[1]));
      
      for (int i=nh+1;i<TRACKPOINTS;i++)
        {
          rp[i]=0.;
        }
      
      imx=(int)(8*d);

      for (int i=0;i<=imx;i++)
        {
          phi=2.*PI/(MYFLOAT)imx*(MYFLOAT)i;
          phd=atanp(cos(phi)*sin(tho),cos(tho));
          phd=fmod(phd,PI);
          ii=0;
          tst=0;
	  
          do
            {
              php=phd+(MYFLOAT)ii*PI;
              nr=php/h;
              ni=(int)nr;
              
              if ((MYFLOAT)ni<nh)
                {
                  r=(rp[ni+1]-rp[ni])*(nr-ni*1.)+rp[ni];
                }
              else
                {
                  r=rp[ni];
                }
              
              if ((r<=re)&&(r>=ri))
                {
                  tst=1;
                  // impact(d,phi,dim,r,b,tho,m,AZP,AFP,q,db,h,bss,raie);
                  // Copie de la procedure ici...
                  MYFLOAT xe,ye;
                  int xi,yi;
                  MYFLOAT flx,rf;
                  xe=d*sin(phi);
                  ye=-d*cos(phi);
                  
                  xi=(int)xe+dim/2;
                  yi=(int)ye+dim/2;
                  
                  rf=decalage_spectral(r,b,phi,tho,m);
                  
                  if (raie==0)
                    {
                      // bss=1.e19;
                      flx=spectre_cn(rf,b,db,h,r,m,bss);
                    }
                  else
                    {
                      // bss=2.;
                      flx=spectre(rf,q,b,db,h,r,m,bss);
                    }
                                    
                  if (AZP[xi+dim*yi]==0.)
                    {
                      AZP[xi+dim*yi]=1./rf;
                    }
                  
                  if (AFP[xi+dim*yi]==0.)
                    {
                      AFP[xi+dim*yi]=flx;
                    }
                  
                }
              
              ii++;
            } while ((ii<=2)&&(tst==0));
        }
    });
  });
  sycl::host_accessor AZP{IZP};
  sycl::host_accessor AFP{IFP};

  // Set stop timer
  gettimeofday(&tv2, NULL);
  mtv2=clock()*1000/CLOCKS_PER_SEC;
  epoch2=time(NULL);
  
  elapsed=(MYFLOAT)((tv2.tv_sec-tv1.tv_sec) * 1000000L +
		    (tv2.tv_usec-tv1.tv_usec))/1000000;
  cputime=(MYFLOAT)((mtv2-mtv1)/1000.);  
  epoch=(MYFLOAT)(epoch2-epoch1);  

  fmx=fp[0];
  zmx=zp[0];

  for (int i=0;i<dim;i++) for (int j=0;j<dim;j++)
    {
      if (fmx<fp[i+dim*j])
	{
	  fimx=i;
	  fjmx=j;
	  fmx=fp[i+dim*j];
	}
      
      if (zmx<zp[i+dim*j])
	{
	  zimx=i;
	  zjmx=j;
	  zmx=zp[i+dim*j];
	}
    }

  printf("\nElapsed Time : %lf",(double)elapsed);
  printf("\nCPU Time : %lf",(double)cputime);
  printf("\nEpoch Time : %lf",(double)epoch);
  printf("\nZ max @(%.6f,%.6f) : %.6f",
	 (float)zimx/(float)dim-0.5,0.5-(float)zjmx/(float)dim,zmx);
  printf("\nFlux max @(%.6f,%.6f) : %.6f\n\n",
	 (float)fimx/(float)dim-0.5,0.5-(float)fjmx/(float)dim,fmx);

  // If input parameters set without output files precised
  if (argc!=7) {
  
    for (int i=0;i<dim;i++)
      for (int j=0;j<dim;j++)
	{
	  zcl=(int)(255/zmx*zp[i+dim*(dim-1-j)]);
	  fcl=(int)(255/fmx*fp[i+dim*(dim-1-j)]);
	  
	  if (strcmp(argv[6],"NEGATIVE")==0)
	    {
	      if (zcl>255)
		{
		  izp[i+dim*j]=0;
		}
	      else
		{
		  izp[i+dim*j]=255-zcl;
		}
	      
	      if (fcl>255)
		{
		  ifp[i+dim*j]=0;
		}
	      else
		{
		  ifp[i+dim*j]=255-fcl;
		}
	  
	    }
	  else
	    {
	      if (zcl>255)
		{
		  izp[i+dim*j]=255;
		}
	      else
		{
		  izp[i+dim*j]=zcl;
		}
	      
	      if (fcl>255)
		{
		  ifp[i+dim*j]=255;
		}
	      else
		{
		  ifp[i+dim*j]=fcl;
		}
	      
	    }
	
	}
    
    if (argc==9)
      {
	  sauvegarde_pgm(argv[7],ifp,dim);
	  sauvegarde_pgm(argv[8],izp,dim);
      }
    else
      {
	sauvegarde_pgm("z.pgm",izp,dim);
	sauvegarde_pgm("flux.pgm",ifp,dim);
      }
  }
  else
    {
      printf("No output file precised, useful for benchmarks...\n\n");
    }

  free(zp);
  free(fp);
  
  free(izp);
  free(ifp);

}


