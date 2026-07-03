// NNFS_Extreme.cpp : Defines the entry point for the application.
#include "Value.hpp"

int main() {

	auto a = newValue(3);
	auto c = a - 7.0f;
	auto d = c * 3;
	auto e = d / 3;
	auto f = e->pow(2);
	auto g = f->operator-();
	a->info(); c->info(); d->info(); e->info(), f->info(); g->info();
	return 0;

}