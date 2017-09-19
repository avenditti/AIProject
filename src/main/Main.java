package main;

public class Main {
	public static void main(String args[]) {
		Controller c = new Controller();
		double[] s = c.trainNetwork(TestInformation.inputs[0],TestInformation.desiredOutputs[0]);
		for(double p : s) {
			System.out.println(p);
		}
	}
}
