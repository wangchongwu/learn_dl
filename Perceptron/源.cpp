#include "Perceptron.h"

float f(float x)
{
	return x > 0 ? 1 : 0;
}
int main()
{
	Perceptron pc(2);
	pc.activator = &f;
	pc.stateInfo();
	vector<vector<float>> vecs = { {1,1},{0,0},{1,0},{0,1} };
	vector<float> labels = { 1,0,1,1 };
	pc.train(vecs, labels, 8, 0.1);

	cout << "1 and 1 = " << pc.predict({ 1,1 }) << endl;
	cout << "0 and 0 = " << pc.predict({ 0,0 }) << endl;
	cout << "1 and 0 = " << pc.predict({ 1,0 }) << endl;
	cout << "0 and 1 = " << pc.predict({ 0,1 }) << endl;
	return 0;
}