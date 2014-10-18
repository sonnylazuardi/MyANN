import java.util.List;

import weka.core.Instance;
import weka.core.Instances;

public class Neuron {
	private Integer activationFunction = 1; // 1. sigmoid, 2. step, 3. sign
	private double[] inputWeight;
	private double[][] input;
	private int idx_instance;
	private double threshold; // buat threshold step

	public Neuron(Instances data) {
		this.threshold = 0.5;
		idx_instance = 0;
		input = new double[100][100];
		for (int i = 0; i < data.numInstances(); i++) {
			for (int j = 0; j < data.get(i).numAttributes()-1; j++) {
				input[i][j] = data.get(i).value(j);
				//System.out.print(data.get(i).value(j));
				System.out.print(input[i][j]+" -- ");
			}
			System.out.println();
		}
	}

	public double summingFunction() {
		double sum = 0;
		for (int i = 0; i < inputWeight.length; i++) {
			sum += inputWeight[i] * input[i][idx_instance];
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

	public double output() {
		double output = 0;
		switch (activationFunction) {
		case 1:
			output = sigmoid(summingFunction());
			break;
		case 2:
			output = step(summingFunction());
			break;
		case 3:
			output = sign(summingFunction());
			break;
		default:
			break;
		}
		return output;
	}
}
