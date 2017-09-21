package main;

public class Neuron {

	private double threshhold;
	private Neuron[] prevNeurons;
	double[] weights;
	private int neuronNumber;
	private double previousValue;
	double[] oldWeights;
	private double previousThreshhold;

	/*
	 * Next two constructors create random connection values to all the other neurons in the next layer
	 *
	 */
	public Neuron(Neuron[] prevNeurons, int totalNextLayer, int neuronNumber) {
		this.prevNeurons = prevNeurons;
		this.neuronNumber = neuronNumber;
		weights = new double[totalNextLayer];
		oldWeights = new double[totalNextLayer];
		for(int i = 0; i < weights.length; i++) {
			//Create a random connection value between -2.4/Fi and 2.4/Fi
			//where Fi is the total number of input neurons to this neuron
			double f = (Math.random() * (2.4/prevNeurons.length - -2.4/prevNeurons.length)) + -2.4/prevNeurons.length;
			weights[i] = f;
			oldWeights[i] = f;
		}
		threshhold = (Math.random() * (2.4/prevNeurons.length - -2.4/prevNeurons.length)) + -2.4/prevNeurons.length;
	}

	/*
	 * Constructor for input layer neurons
	 */

	public Neuron(int totalNextLayer, int neuronNumber,int totalInputNeurons) {
		weights = new double[totalNextLayer];
		oldWeights = new double[totalNextLayer];
		this.neuronNumber = neuronNumber;
		for(int i = 0; i < weights.length; i++) {
			//Create a random connection value between -2.4/Fi and 2.4/Fi
			//where Fi is the total number of input neurons to this neuron
			double f = (Math.random() * (2.4/totalInputNeurons - -2.4/totalInputNeurons)) + -2.4/totalInputNeurons;
			weights[i] = f;
			oldWeights[i] = f;
		}
		threshhold = (Math.random() * (2.4/totalInputNeurons - -2.4/totalInputNeurons)) + -2.4/totalInputNeurons;
	}

	/*
	 * Constructor for output layer neurons
	 */

	public Neuron(Neuron[] prevNeurons, int neuronNumber) {
		weights = new double[0];
		this.neuronNumber = neuronNumber;
		this.prevNeurons = prevNeurons;
		threshhold = (Math.random() * (2.4/prevNeurons.length - -2.4/prevNeurons.length)) + -2.4/prevNeurons.length;
	}

	public double getWeight(int i) {
		return weights[i];
	}

	public double activationFunction(double x) {
		return 1/(1+Math.pow(Math.E,(-1 * x)));
	}

	public double fire(double[] data) {
		double output = 0;
		if(prevNeurons != null) {
			for(int i = 0; i < prevNeurons.length; i++) {
				output += data[i] * prevNeurons[i].getWeight(neuronNumber) - threshhold;
			}
		} else {
			output = data[neuronNumber] - threshhold;
		}
		return previousValue = activationFunction(output);
	}

	public String toString() {
		String s = "";
		for(int i = 0; i < weights.length; i++) {
			s += i + " " + weights[i] + "\n";
		}
		return s + neuronNumber + " " + threshhold + "\n";
	}

	public void adjustNeuronWeight(double[] d, double learningRate, double previousValue) {
		double weightChange = learningRate * previousValue;
		for(int i = 0; i < weights.length; i++) {
			oldWeights[i] = weights[i];
			weights[i] += weightChange * d[i];
		}
	}

	public double getPreviousValue() {
		return previousValue;
	}

	public double getThreshhold() {
		return threshhold;
	}

	public Neuron[] getPrevNeurons() {
		return prevNeurons;
	}


	public double getPreviousThreshhold() {
		return previousThreshhold;
	}

	public void setThreshhold(double d) {
		this.threshhold = d;
	}

}
