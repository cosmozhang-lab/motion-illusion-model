#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#define ABS(x) (((x) > 0) ? (x) : (-x))

#define EPS (1.1920929e-07)

#define RAND_MAX (-1U)

/**
 * Compare two floats within a toleratable error
 * range. Returns true if they are almostly equal.
 */
inline bool almost_equal(float a, float b) {
  float delta = a - b;
  if (delta > -EPS && delta < EPS) return true;
  else return false;
}

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
    if (t < EPS && t > -EPS) break;
  }
  res = res * 2 + nln10 * ln10;
  return res;
}

/**
 * Calculate natural exponential
 * We use: exp(x) = lim[n->inf]{ (1+x/n)^n }
 * Here we find a n in {1,2,4,8,...} that make x/n very small,
 * and we can then square (1+x/n) again and again for log[2]{n}
 * times to get the result.
 * See: http://blog.csdn.net/aaronand/article/details/50269131
 */
inline float expf(float x) {
  int n = 1;
  int nn = 0; // log[2]{n}
  float absx = ABS(x);
  while (absx / n > 0.001) {
    n *= 2;
    nn += 1;
  }
  float res = 1.0 + x / n;
  for (int i = 0; i < nn; i++) {
    res = res * res;
  }
  return res;
}






