import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class Main {
	
	public static void main (String[] args) throws Exception {
		DataSource source = new DataSource("D:/and.arff");
		Instances data = source.getDataSet();
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		
		MyNN my_nn = new MyNN();
		
		my_nn.PTR(data);
	}

}
