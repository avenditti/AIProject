package main;

public class NeuronLayer {

	private Neuron[] nes; // Neurons in the current layer
	double[] prevOutput;
	NeuronLayer pl;

	/*
	 * n = number of neurons in the current layer
	 * pl = neurons in the preceding layer
	 * nl = number of neurons in the next layer
	 *
	 */

	public NeuronLayer(int n, NeuronLayer pl, int nl) {
		nes = new Neuron[n];
		this.pl = pl;
		for(int i = 0; i < n; i++) {
			nes[i] = new Neuron(pl.getNeurons(), nl, i);
		}
	}

	/*
	 * Constructor for the input layer
	 */

	public NeuronLayer(int n, int nl) {
		nes = new Neuron[n];
		for(int i = 0; i < n; i++) {
			nes[i] = new Neuron(nl, i, n);
		}
	}

	/*
	 * Constructor for the output layer
	 */
	public NeuronLayer(NeuronLayer pl, int n) {
		nes = new Neuron[n];
		this.pl = pl;
		for(int i = 0; i < n; i++) {
			nes[i] = new Neuron(pl.getNeurons(), i);
		}
	}

	public NeuronLayer(Neuron[] neurons) {
		this.nes = neurons;
	}
	public NeuronLayer(Neuron[] neurons, NeuronLayer pl) {
		this.nes = neurons;
		this.pl = pl;
	}

	public Neuron[] getNeurons() {
		return nes;
	}

	public int getSize() {
		return nes.length;
	}

	public double[] processData(double[] data) {
		double[] output;
		output = new double[nes.length];
		for(int i = 0; i < nes.length; i++) {
			output[i] = nes[i].fire(data);
		}
		prevOutput = output;
		return output;
	}

	public String toString() {
		String s = "";
		for(Neuron n : nes) {
			s += n.toString() + "\n";
		}
		return s;
	}
}
