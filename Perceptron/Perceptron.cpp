#include "Perceptron.h"



Perceptron::Perceptron(int num)
{
	//感知机初始化
	input_num = num;
	this->weights.resize(num);
	this->bias = 0;
}


Perceptron::~Perceptron()
{
}

void Perceptron::stateInfo()
{
	cout << "weights:";
	for (int i = 0; i < input_num; i++)
	{
		cout << weights[i] << "  ";
	}
	cout << endl;
	cout << "bias:" << bias<<endl;
}

float Perceptron::predict(vector<float> inputX)
{
	float sumV = 0;
	for (int i = 0; i < input_num; i++)
	{
		sumV += inputX[i] * weights[i];
	}
	sumV += bias;

	return activator(sumV);//回调
}

void Perceptron::update_weights(vector<float> inputX, float output, float label)
{
	float delta = label - output;
	for (int i = 0;i<input_num;i++)
	{
		this->weights[i] += inputX[i] * delta*this->rate;
		this->bias += this->rate*delta;
	}
}

void Perceptron::one_iteration(vector< vector<float> > inputXs, vector<float> labels)
{
	for (int i = 0; i < inputXs.size(); i++)
	{
		float output = predict(inputXs[i]);
		update_weights(inputXs[i], output, labels[i]);
	}
}

void Perceptron::train(vector<vector<float>> inputXs, vector<float> labels, int iteration,float rate)
{
	this->rate = rate;
	for (int i = 0; i < iteration; i++)
	{
		this->stateInfo();
		one_iteration(inputXs, labels);
	}
}
