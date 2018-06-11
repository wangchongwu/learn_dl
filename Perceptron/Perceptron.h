#pragma once
#include <vector>
#include <iostream>
using namespace std;


class Perceptron
{
public:
	Perceptron(int num);
	~Perceptron();

	void stateInfo();
	float predict(vector<float> input);//输入响应
	void update_weights(vector<float> input, float output, float label);//更新权值
	void one_iteration(vector< vector<float> > inputXs, vector<float> labels);//迭代一次
	void train(vector< vector<float> > inputXs, vector<float> labels,int iteration, float rate);//带入所有数据进行训练
public:
	float(*activator)(float X);//激活函数指针

	int input_num;//神经元个数
	float bias;//偏置
	vector<float> weights; //权值

	float rate;//学习率

};

