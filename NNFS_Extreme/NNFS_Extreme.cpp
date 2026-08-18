#include "examples/standard_mnist.hpp"
#include "examples/inverted_mnist.hpp"
#include "examples/model_export_n_import.hpp"
#include "examples/import_model.hpp"
#include "utils.hpp"

int main() 
{
	// warmup();
	standard_mnist();
	// inverted_mnist();
	// import_model("data/output/mnist_inverted_10-128-256-512-784_ep20_lr15p000.bin");
	// model_export_n_import(10, 32, 9.0F);
	
	return 0;
}