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
	float predict(vector<float> input);//������Ӧ
	void update_weights(vector<float> input, float output, float label);//����Ȩֵ
	void one_iteration(vector< vector<float> > inputXs, vector<float> labels);//����һ��
	void train(vector< vector<float> > inputXs, vector<float> labels,int iteration, float rate);//�����������ݽ���ѵ��
public:
	float(*activator)(float X);//�����ָ��

	int input_num;//��Ԫ����
	float bias;//ƫ��
	vector<float> weights; //Ȩֵ

	float rate;//ѧϰ��

};

