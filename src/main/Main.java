package main;

public class Main {
	public static void main(String args[]) {
		Controller c = new Controller();
		System.out.println(c.trainNetwork(TestInformation.inputs,TestInformation.desiredOutputs));
		boolean[] s = c.testNetwork(TestInformation.inputs2, TestInformation.desiredOutputs2);
		for(boolean d : s) {
			System.out.println(d);
		}
		System.out.println(s[s.length - 1]);
	}
}
