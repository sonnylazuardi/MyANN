import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;


public class MyNN implements Classifier {
	private Double learningRate;
	private Double momentum;
	private Integer hiddenLayers;
	private Double initialWeight;
	private Integer topologi = 1; // 1. Perceptron, 2. MLP-backprop
	
	private Neuron neurons; // ngetest PTR
	
	public MyNN() {
		
	}
	
	public void PTR(Instances data) {
		double w1 = Math.random();
		double w2 = Math.random();
		double[] weight = {w1, w2};
		System.out.println("Weight : "+w1+"--"+w2);
		neurons = new Neuron(data, weight, 0.1);
		neurons.train();
		System.out.println("Hasil : "+neurons.getClassify(0,1));
		
	}
	@Override
	public void buildClassifier(Instances data) throws Exception {
		
		
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
	
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}
	

	
	
	
}
