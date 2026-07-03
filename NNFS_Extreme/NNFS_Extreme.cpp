// NNFS_Extreme.cpp : Defines the entry point for the application.
//

#include "NNFS_Extreme.h"
#include "matrix.h"

using namespace std;

int main()
{
	cout << "Hello CMake Bro are you good?." << endl;
	
	auto m1 = mat_random_int_range(30, 30, 1, 9);
	auto m2 = mat_random_int_range(30, 30, 1, 9);

	auto m3 = m1 * m2;
	printMatrix(m3);

	return 0;
}
