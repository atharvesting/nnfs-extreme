#include "examples/standard_mnist.hpp"
#include "examples/inverted_mnist.hpp"
#include "examples/model_export_n_import.hpp"
#include "utils.hpp"

int main() 
{
	// warmup();
	// standard_mnist();
	// inverted_mnist();
	model_export_n_import(10);
	
	return 0;
}