import java.util.List;

import weka.core.Instance;
import weka.core.Instances;

public class Neuron {
	private Integer activationFunction = 3; // 1. sigmoid, 2. step, 3. sign
	private double[] inputWeight;
	private double[][] input;
	private double[] desire;
	private double output;
	private int idx_instance;
	private double threshold; // buat threshold step
	private double rate;
	private double bias = 0;
	private int epoch;
	private double min_error = 0.05;

	public Neuron(Instances data, double[] weight, double rate) {
		this.epoch = 0;
		this.threshold = 0.6;
		this.rate = rate;
		idx_instance = 0;
		input = new double[100][100];
		desire = new double[100];
		inputWeight = weight;
		
		//masukin input dan desire input
		for (int i = 0; i < data.numInstances(); i++) {
			for (int j = 0; j < data.get(i).numAttributes(); j++) {
				if (j == data.get(i).numAttributes()-1 ) {
					desire[i] = data.get(i).value(j);
				} else 
					input[i][j] = data.get(i).value(j);
			}
		}
	}

	public double summingFunction() {
		double sum = 0;
		for (int i = 0; i < inputWeight.length; i++) {
			sum += inputWeight[i] * input[idx_instance][i] + bias;
		}
		return sum;
	}

	public double sigmoid(double temp) {
		return (1 / (1 + Math.exp(-temp)));
	}

	public double step(double temp) {
		if (temp >= threshold) {
			return 1;
		} else {
			return 0;
		}
	}

	public double sign(double temp) {
		if (temp < 0) {
			return -1;
		} else {
			return 1;
		}
	}

	public void getOutput() {
		switch (activationFunction) {
		case 1:
			if (sigmoid(summingFunction()) > 45)
				output = 1;
			else
				output = 0;
			break;
		case 2:
			output = step(summingFunction());
			break;
		case 3:
			if (sign(summingFunction()) == -1.0)
				output = 0;
			else
				output = 1;
			break;
		default:
			break;
		}
	}
	
	public double getClassify(double i, double j) {
		//System.out.println("i j : "+i+"--"+j);
		//System.out.println("weight1 weight2 : "+inputWeight[0]+"--"+inputWeight[1]);
		double sum = inputWeight[0] * i + inputWeight[1] * j;
		switch (activationFunction) {
		case 1:
			if (summingFunction() > 1)
				sum = 1;
			else if (summingFunction() < -1)
				sum = 0;
			else {
				System.out.println("yang ini");
				return sigmoid(sum);
			}
			break;
		case 2:
			sum = step(sum);
			break;
		case 3:
			sum = sign(sum);
			break;
		default:
			break;
		}
		return sum;
	}
	public void updateWeight() {
		double deltaW;
		//boolean run = false;
		for (int i = 0; i < inputWeight.length; i++) {
			deltaW = rate * (desire[idx_instance] - output)*input[idx_instance][i];
			inputWeight[i] += deltaW;
//			System.out.println("deltaw : "+deltaW);
//			if (Math.abs(deltaW) >= 0.1) //ga konvergen
//			{
//				weight[i] += deltaW;
//				run = true;
//			}
		}
//		/return run;
	}
	
	//calculate mean square error
	public double calculateMSE(){
		double result = 0;
		double error_rate = 0;
		
		for (int i = 0; i < input.length; i++) {
			double classification = getClassify(input[i][0], input[i][1]);
			error_rate = desire[i] - classification;
			error_rate = error_rate * error_rate;
			result += error_rate;
		}
		//divided by attribute number
		result = result / 2;
		System.out.println("MSE : "+result+"== iterasi ke"+epoch);
		return result;
	}
	
	public void nextInstance() {
		if (idx_instance > 3) {
			idx_instance = 0;
			epoch++;
		} else {
			idx_instance++;
		}
	}
	
	public void train() {
		double mse = 0;
		do{
			getOutput();
			updateWeight();
			//System.out.println("Weight : "+inputWeight[0]+"--"+inputWeight[1]+" input"+input[idx_instance][0]+"=="+input[idx_instance][1]);
			mse = calculateMSE();
			nextInstance();
		}
		while((epoch < 100) && (mse > min_error));
	}
}