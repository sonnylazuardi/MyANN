import java.util.List;

import weka.core.Instance;
import weka.core.Instances;

public class Neuron {
	private Integer activationFunction = 2; // 1. sigmoid, 2. step, 3. sign
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
	private Integer learningRule = 3; // 1. threshold, 2. batch gradient 3. delta rule
	private double[] gradientDescentWeight;
	
	public Neuron(Instances data, double[] weight, double rate) {
		this.epoch = 0;
		this.threshold = 0.6;
		this.rate = rate;
		idx_instance = 0;
		input = new double[100][100];
		desire = new double[100];
		inputWeight = weight;
		gradientDescentWeight = new double[2];
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
		if (temp >= 0.6) {
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
		if (learningRule == 1) { //1.threshold
			//System.out.println("simple perceptron :");
			for (int i = 0; i < inputWeight.length; i++) {
				deltaW = rate * (desire[idx_instance] - output)*input[idx_instance][i];
				inputWeight[i] += deltaW;
				//print current weight
				System.out.print(inputWeight[i]+" ");
			}
			System.out.println();
		}
		else if (learningRule == 3) { //3.delta rule
			//System.out.println("delta rule :");
			for (int i = 0; i < inputWeight.length; i++) {
				deltaW = rate * (desire[idx_instance] - summingFunction())*input[idx_instance][i];
				inputWeight[i] += deltaW;
				//print current weight
				System.out.print(inputWeight[i]+" ");
			}
			System.out.println();
		}
		else{ //2.batch gradient descent
			//System.out.println("batch gradient descent :");
			for (int i = 0; i < inputWeight.length; i++) {
				deltaW = rate * gradientDescentWeight[i];
				inputWeight[i] += deltaW;
				//print current weight
				System.out.print(inputWeight[i]+" ");
			}
			System.out.println();
		}
		
	}
	
	//use gradient descent for weight updating
	/*public void updateWeightGradDescent(){
		double deltaW;
		for (int i = 0; i < inputWeight.length; i++) {
			
		}
	}*/
	
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
		//System.out.println("MSE : "+result+"== iterasi ke"+epoch);
		return result;
	}
	
	public void nextInstance() {
		if (idx_instance >= 3) {
			if(learningRule == 2){
				updateWeight();
			}
			idx_instance = 0;
			epoch++;
			for(int i = 0; i < 60; i ++) System.out.print("-");
			System.out.println("\nEpoch ke-"+epoch);
		} else {
			idx_instance++;
		}
	}
	
	public void train() {
		for(int i = 0; i < 60; i ++) System.out.print("-");
		System.out.println("\nEpoch ke-"+epoch);
		double mse = 0;
		do{
			if(learningRule != 2){
				getOutput();
				updateWeight();
				//System.out.println("Weight : "+inputWeight[0]+"--"+inputWeight[1]+" input"+input[idx_instance][0]+"=="+input[idx_instance][1]);
				mse = calculateMSE();
				nextInstance();
			}
			else{
				getOutput();
				for (int i = 0; i < gradientDescentWeight.length; i++) {
					gradientDescentWeight[i] += (desire[idx_instance] - summingFunction())*input[idx_instance][i];
				}
				mse = calculateMSE();
				nextInstance();
			}
		}
		while((epoch < 2000) && (mse > min_error));
		//print final weight
		System.out.println("\nFinal Weight :");
		for(int i = 0; i < inputWeight.length; i++){
			System.out.print(inputWeight[i]+" ");
		}
		System.out.println();
	}
}