/* Direct search for integer sharpness witnesses of Theorem A at n = 4 (independent of the pencil):
 * all pairs m != m' of 4-multisets of positive integers <= N with equal R = sum 1/m_i,
 * P_1 = sum m_i, P_3 = sum m_i^3 (equivalently: equal first three heat coefficients, [Sig] Lemma 4
 * with n = n' = 4, g = g' = 0).  Exact int64 arithmetic: P3 <= 4N^3, e3 <= 4N^3, e4 <= N^4 (N <= 1000).
 * For each sum s = P_1 the multisets with that sum are sorted by (P3, e3/g, e4/g), g = gcd(e3,e4)
 * (R = e3/e4).  Every collision is printed: "s P3 a b c d | a' b' c' d'".
 * Checkpoint: after each s, the line "#done s" is printed and flushed (output is the checkpoint).
 * Usage: check_direct4 N [s_start]
 */
#include <stdio.h>
#include <stdlib.h>
typedef struct { long long p3, r1, r2; int x[4]; } rec;
static long long gcdll(long long a, long long b){ if(a<0)a=-a; if(b<0)b=-b; while(b){long long t=a%b;a=b;b=t;} return a; }
static int cmp(const void*p,const void*q){ const rec*a=p,*b=q;
  if(a->p3!=b->p3) return a->p3<b->p3?-1:1; if(a->r1!=b->r1) return a->r1<b->r1?-1:1;
  if(a->r2!=b->r2) return a->r2<b->r2?-1:1; return 0; }
int main(int argc,char**argv){
  int N = atoi(argv[1]); int s0 = argc>2 ? atoi(argv[2]) : 4;
  if(N>1000) return 2;
  size_t cap = 1<<16; rec *R = malloc(cap*sizeof(rec));
  long long total=0, coll=0;
  for(int s=s0; s<=4*N; s++){
    size_t n=0;
    for(int a=1; 4*a<=s; a++) for(int b=a; a+3*b<=s; b++) for(int c=b; a+b+2*c<=s; c++){
      int d = s-a-b-c; if(d<c || d>N) continue;
      long long A=a,B=b,C=c,D=d;
      long long e3 = A*B*C+A*B*D+A*C*D+B*C*D, e4 = A*B*C*D, g = gcdll(e3,e4);
      if(n==cap){ cap*=2; R=realloc(R,cap*sizeof(rec)); }
      R[n].p3 = A*A*A+B*B*B+C*C*C+D*D*D; R[n].r1=e3/g; R[n].r2=e4/g;
      R[n].x[0]=a;R[n].x[1]=b;R[n].x[2]=c;R[n].x[3]=d; n++;
    }
    total += n;
    qsort(R,n,sizeof(rec),cmp);
    for(size_t i=0;i+1<n;i++) for(size_t j=i+1;j<n && cmp(&R[i],&R[j])==0;j++){
      coll++;
      printf("%d %lld %d %d %d %d | %d %d %d %d\n",s,R[i].p3,R[i].x[0],R[i].x[1],R[i].x[2],R[i].x[3],R[j].x[0],R[j].x[1],R[j].x[2],R[j].x[3]);
    }
    printf("#done %d\n",s); fflush(stdout);
  }
  printf("# N=%d multisets=%lld collisions=%lld\n",N,total,coll);
  return 0;
}
