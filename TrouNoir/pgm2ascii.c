#include <stdio.h>
#include <stdlib.h>

#define MYSIZE 24
#define UCHAR_MAX 255

void sauvegarde_pgm(char nom[MYSIZE],unsigned int *image,int dim)
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

void lecture_pgm(char nom[24])
{
  
  FILE            *entree;
  unsigned long   i,j;
  char MyType[MYSIZE],g,c;
  unsigned int MyX,MyY,MyDyn,p;
  char map[10] = " .,:;ox%#@";
  
  entree=fopen(nom,"r");
  fscanf(entree," %s\n%d %d\n%d\n",MyType,&MyX,&MyY,&MyDyn);

  printf("Image de type %s, de taille %d*%d avec dynamique %d\n",
	 MyType,MyX,MyY,MyDyn);

  for (j=0;j<MyY;j++)
    {
      for (i=0;i<MyX;i++)
	{
	  g=fgetc(entree);
	  p=2*(char) ((g < 0) ? (UCHAR_MAX - g) : g);
	  c=map[p*10/256];
	  printf("%c%c",c,c);
	}
      printf("\n");
    }
  
  fclose(entree);
}

int main(int argc,char **argv)
{
  lecture_pgm(argv[1]);
}
