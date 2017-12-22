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
unsigned int rand(unsigned int seed) {
  return 1103515245U * seed + 12345U;
}