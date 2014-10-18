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
		neurons = new Neuron(data);
	}
	@Override
	public void buildClassifier(Instances data) throws Exception {
		
		
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
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
