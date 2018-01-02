#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define MIN(x,y) (((x) < (y)) ? (x) : (y))

#define RAND_MAX (-1U)

/**
 * Random generator
 * Algorithm: LCG (linear congruential generator)
 * > iterates: x[i+1] = (a * x[i] + c) % m
 * Some implementations are:
 *      _________________________________________________________________
 *     | Implementation               |          a |          c | m      |
 *     |------------------------------|------------|------------|--------|
 *     | Numerical Recipes            |    1664525 | 1013904223 | 2^32   |
 *     | Borland C/C++                |   22695477 |          1 | 2^32   |
 *     | glibc (used by GCC)          | 1103515245 |      12345 | 2^32   |
 *     | ANSI C                       | 1103515245 |      12345 | 2^32   |
 *     | Borland Delphi               |  134775813 |          1 | 2^32   |
 *     | Virtual Pascal               |  134775813 |          1 | 2^32   |
 *     | Microsoft Visual/Quick C/C++ |     214013 |    2531011 | 2^32   |
 *     | Apple CarbonLib              |      16807 |          0 | 2^32-1 |
 *     |______________________________|____________|____________|________|
 *     * Ansi C: Watcom, Digital Mars, CodeWarrior, IBM VisualAge C/C++
 * We are using `glibc` parameters
 */
inline unsigned int rand(unsigned int seed) {
  return 1103515245U * seed + 12345U;
}

/**
 * Calculate natural logarithm
 * We use Taylor expansion of log to calculate:
 *     log(1+x) = x - x^2/2 + x^3/3 - ... + o(x^n)
 * See: http://blog.csdn.net/aaronand/article/details/50269131
 */
inline float logf(float x) {
  const float ln10 = 2.30258509299404609679094392;
  const float eps = 0.000000000000000000000000001;
  int nln10 = 0;
  while (x > 10.0) {
    x = x / 10.0;
    nln10++;
  }
  float t = (x-1)/(x+1);
  float t02 = t*t;
  float res = 0.0;
  int k = 1;
  while (true) {
    res += t / k;
    k += 2;
    t = t * t02;
    if (t < eps && t > -eps) break;
  }
  res = res * 2 + nln10 * ln10;
  return res;
}




