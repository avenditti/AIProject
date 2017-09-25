package main;

public class Neuron {

	private double threshhold;
	private Neuron[] prevNeurons;
	double[] weights;
	private double neuronNumber;
	private double previousValue;
	double[] oldWeights;
	private double previousThreshhold;

	/*
	 * Next two constructors create random connection values to all the other neurons in the next layer
	 *
	 */
	public Neuron(Neuron[] prevNeurons, int totalNextLayer, double... neuronNumber) {
		this.prevNeurons = prevNeurons;
		this.neuronNumber = neuronNumber[0];
		weights = new double[totalNextLayer];
		oldWeights = new double[totalNextLayer];
		if(neuronNumber.length > 1) {
			for(int i = 0; i < weights.length; i++) {
				//Create a weight array with the given constraints
				double f = neuronNumber[i + 2];
				weights[i] = f;
				oldWeights[i] = f;
			}
			threshhold = neuronNumber[1];
		} else {
			for(int i = 0; i < weights.length; i++) {
				//Create a random connection value between -2.4/Fi and 2.4/Fi
				//where Fi is the total number of input neurons to this neuron
				double f = (Math.random() * (2.4/prevNeurons.length - -2.4/prevNeurons.length)) + -2.4/prevNeurons.length;
				weights[i] = f;
				oldWeights[i] = f;
			}
			threshhold = (Math.random() * (2.4/prevNeurons.length - -2.4/prevNeurons.length)) + -2.4/prevNeurons.length;
		}
	}

	/*
	 * Constructor for input layer neurons
	 */

	public Neuron(int totalNextLayer,int totalInputNeurons, double... neuronNumber) {
		weights = new double[totalNextLayer];
		oldWeights = new double[totalNextLayer];
		this.neuronNumber = neuronNumber[0];
		if(neuronNumber.length > 1) {
			for(int i = 0; i < weights.length; i++) {
				//Create a weight array with the given constraints
				double f = neuronNumber[i + 1];
				weights[i] = f;
				oldWeights[i] = f;
			}
			threshhold = 0;
		} else {
			for(int i = 0; i < weights.length; i++) {
				//Create a random connection value between -2.4/Fi and 2.4/Fi
				//where Fi is the total number of input neurons to this neuron
				double f = (Math.random() * (2.4/totalInputNeurons - -2.4/totalInputNeurons)) + -2.4/totalInputNeurons;
				weights[i] = f;
				oldWeights[i] = f;
			}
			threshhold = 0;//(Math.random() * (2.4/totalInputNeurons - -2.4/totalInputNeurons)) + -2.4/totalInputNeurons;
		}

	}

	/*
	 * Constructor for output layer neurons
	 */

	public Neuron(Neuron[] prevNeurons, double... neuronNumber) {
		weights = new double[0];
		this.neuronNumber = neuronNumber[0];
		this.prevNeurons = prevNeurons;
		if(neuronNumber.length > 1) {
			threshhold = neuronNumber[1];
		} else {
			threshhold = (Math.random() * (2.4/prevNeurons.length - -2.4/prevNeurons.length)) + -2.4/prevNeurons.length;
		}
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
				output += data[i] * prevNeurons[i].getWeight((int)neuronNumber);
			}
		} else {
			return previousValue = data[(int)neuronNumber];
		}
		output -= threshhold;
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
