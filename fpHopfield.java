/* Authors: Chaise Brown & Will Carhart
 * Due Date: 04/20/17
 * Summary: Implements a Hopfield Neural Network.
 */

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;
import java.util.StringTokenizer;

public class fpHopfield {
	public static void main(String[] args) {
		// File I/O Variables
		Scanner input = new Scanner(System.in);
		BufferedReader br = null;
		String imageTrainingFile = null;
		String imageTestingFile = null;
		String savedWeightsFile = null;
		String resultsFile = null;
		double steepness, errorThreshold;

		System.out.println("Welcome to my Hopfield Neural Network!");
		while (true) {
			displayUserMenu();
			System.out
					.println("Please enter a command from the user menu above: ");
			int command = input.nextInt();
			String dummy = input.nextLine();
			switch (command) {
			case 1:
				boolean validTrainingFile = false;
				while (!validTrainingFile) {
					System.out
							.println("Please enter the image training file name (with extension): ");
					imageTrainingFile = input.nextLine();
					try {
						br = new BufferedReader(new FileReader(
								imageTrainingFile));
						validTrainingFile = true;
					} catch (FileNotFoundException e) {
						System.err.print(imageTrainingFile + " not found.\n");
					}
				}
				createKeyFile(imageTrainingFile);
				System.out
						.println("Please enter the file name where the weights will be saved: ");
				savedWeightsFile = input.nextLine();
				train(imageTrainingFile, savedWeightsFile);
				break;
			case 2:
				boolean notTrained = false;
				try {
					br = new BufferedReader(new FileReader("key.txt"));
					if (br.readLine() == null) {
						System.err
								.println("The neural network has not been trained yet. Please train the network before testing");
						notTrained = true;
					}
				} catch (IOException e) {
					System.err
							.println("The neural network has not been trained yet. Please train the network before testing");
					notTrained = true;
				}
				if (notTrained) {
					break;
				}

				boolean validTestingFile = false;
				while (!validTestingFile) {
					System.out
							.println("Please enter the image testing file name (with extension): ");
					imageTestingFile = input.nextLine();
					try {
						br = new BufferedReader(
								new FileReader(imageTestingFile));
						validTestingFile = true;
					} catch (FileNotFoundException e) {
						System.err.print(imageTestingFile + " not found.\n");
					}
				}
				boolean validWeightFile = false;
				while (!validWeightFile) {
					System.out
							.println("Please enter the trained weights settings file name (with extension): ");
					savedWeightsFile = input.nextLine();
					try {
						br = new BufferedReader(
								new FileReader(savedWeightsFile));
						validWeightFile = true;
					} catch (FileNotFoundException e) {
						System.err.print(savedWeightsFile + " not found.\n");
					}
				}
				System.out
						.println("Please enter the file name where the testing results will be saved: ");
				resultsFile = input.nextLine();
				System.out.println("Please enter the steepness parameter.");
				steepness = input.nextDouble();
				System.out.println("Please enter the error threshold.");
				errorThreshold = input.nextDouble();
				test(imageTestingFile, savedWeightsFile, resultsFile,
						steepness, errorThreshold);
				statistics(resultsFile);
				break;
			case 3:
				try {
					br.close();
					input.close();
				} catch (IOException e) {
					System.err.print("Error in I/O");
					System.exit(1);
				}
				System.exit(1);
				break;
			}
		}
	}

	/*
	 * Creates a key file (reference to the training patterns)
	 * 
	 * @param imageTrainingFile the name of the image training file
	 */
	public static void createKeyFile(String imageTrainingFile) {
		// I/O variables
		PrintWriter pw = null;
		BufferedReader br = null;
		String temp;

		try {
			br = new BufferedReader(new FileReader(imageTrainingFile));
			pw = new PrintWriter("key.txt", "UTF-8");

			// transfer from training file to key file
			while ((temp = br.readLine()) != null) {
				pw.write(temp + "\n");
			}

			br.close();

		} catch (FileNotFoundException e) {
			System.err.print("File not found");
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		}
		pw.close();
	}

	/*
	 * Implements the training of the Hopfield NN
	 * 
	 * @param imageTrainingFile the name of the image training file
	 * 
	 * @param savedWeightsFile the name of the saved weights file
	 */
	public static String[] train(String imageTrainingFile,
			String savedWeightsFile) {

		// File I/O Variables
		BufferedReader br = null;
		String toBeTokenized = null;
		String[] lineElements = null;
		int image_dim = 0;

		// training variables
		ArrayList<Double> trainingData = null;

		String[] fileNames = { "matrix0.txt", "matrix1.txt", "matrix2.txt",
				"matrix3.txt", "matrix4.txt", "matrix5.txt", "matrix6.txt",
				"matrix7.txt", "matrix8.txt", "matrix9.txt" };

		System.out.println("Now Training...");

		// read in information about training data
		try {

			br = new BufferedReader(new FileReader(imageTrainingFile));
			toBeTokenized = br.readLine();
			lineElements = toBeTokenized.split(",");
			StringBuilder sb = new StringBuilder(lineElements[0]);
			sb.deleteCharAt(0);
			sb.deleteCharAt(0);
			sb.deleteCharAt(0);
			String newStr = sb.toString();
			image_dim = Integer.parseInt(newStr);
			br.readLine();
			br.readLine();

			double[][] weightMatrix = new double[image_dim][image_dim];
			double[][] averageTargetMatrix = new double[10][image_dim];
			int[] targets = new int[10];
			int target;

			// Training
			while ((toBeTokenized = br.readLine()) != null) {
				lineElements = toBeTokenized.split(",");
				target = Integer.parseInt(lineElements[0]);
				switch (target) {
				case 0:
					targets[0]++;
					break;
				case 1:
					targets[1]++;
					break;
				case 2:
					targets[2]++;
					break;
				case 3:
					targets[3]++;
					break;
				case 4:
					targets[4]++;
					break;
				case 5:
					targets[5]++;
					break;
				case 6:
					targets[6]++;
					break;
				case 7:
					targets[7]++;
					break;
				case 8:
					targets[8]++;
					break;
				case 9:
					targets[9]++;
					break;
				}
				trainingData = new ArrayList<Double>();
				for (int i = 1; i < lineElements.length; i++) {
					// normalize to bipolar vector
					double proportion = (Double.parseDouble(lineElements[i]) / 255);
					double toAdd = ((proportion * 2) - 1);
					trainingData.add(toAdd);
					averageTargetMatrix[target][i - 1] += toAdd;
				}
				// Creating weight matrix
				for (int i = 0; i < image_dim; i++) {
					for (int j = 0; j < image_dim; j++) {
						weightMatrix[i][j] += (trainingData.get(i) * trainingData.get(j));
					}
				}
			}

			// Updating weights on main diagonal once training completes
			int i = 0;
			int j = 0;
			while (i < image_dim && j < image_dim) {
				weightMatrix[i][j] = 0;
				i++;
				j++;
			}
			
			for (int x = 0; x < image_dim; x++) {
				for (int y = 0; y < image_dim; y++) {
					weightMatrix[x][y] /= 500;
				}
			}

			for (int k = 0; k < averageTargetMatrix.length; k++) {
				for (int l = 0; l < averageTargetMatrix[0].length; l++) {
					averageTargetMatrix[k][l] /= targets[k];
				}
			}

			for (int k = 0; k < fileNames.length; k++) {
				PrintWriter pw = new PrintWriter(new FileWriter(fileNames[k]));
				for (int l = 0; l < averageTargetMatrix[0].length; l++) {
					pw.print(averageTargetMatrix[k][l] + " ");
				}
				pw.close();
			}

			System.out.println("Successfully trained.");
			System.out.println("Now saving weights...");

			saveWeights(weightMatrix, savedWeightsFile);

			System.out.println("Weights successfully saved.");

			br.close();
		} catch (IOException e) {
			System.err.print("Error in I/O");
			System.exit(1);
		}

		return fileNames;
	}

	/*
	 * Implements the testing of the trained Hopfield NN
	 * 
	 * @param imageTestingFile the name of the image testing file
	 * 
	 * @param savedWeightsFile the name of the saved weights file
	 * 
	 * @param resultsFile the name of the results file
	 */
	public static void test(String imageTestingFile, String savedWeightsFile,
			String resultsFile, double steepness, double errorThreshold) {
		// File I/O Variables
		BufferedReader br = null;
		String toBeTokenized = null;
		String[] lineElements = null;
		int image_dim = 0;

		System.out.println("Now Testing...");

		// testing variables
		ArrayList<Double> testingData = null;
		ArrayList<Integer> target = new ArrayList<Integer>();
		ArrayList<Double> y = null;
		double[][] weightMatrix = null;
		double yin, yval;
		int index = 0;
		boolean converged;
		int count = 0;

		// read in information about testing data
		try {

			br = new BufferedReader(new FileReader(imageTestingFile));
			toBeTokenized = br.readLine();
			lineElements = toBeTokenized.split(",");
			StringBuilder sb = new StringBuilder(lineElements[0]);
			sb.deleteCharAt(0);
			sb.deleteCharAt(0);
			sb.deleteCharAt(0);
			String newStr = sb.toString();
			image_dim = Integer.parseInt(newStr);
			br.readLine();
			br.readLine();

			weightMatrix = weightMatrix(savedWeightsFile, image_dim);

			// Later used for randomized pattern generator
			ArrayList<Integer> list = new ArrayList<Integer>();
			for (int i = 0; i < image_dim; i++) {
				list.add(new Integer(i));
			}

			// begin testing
			while ((toBeTokenized = br.readLine()) != null) {
				// Read in testing data
				testingData = new ArrayList<Double>();
				lineElements = toBeTokenized.split(",");
				target.add(Integer.parseInt(lineElements[0])); // reads target
																// value
				for (int i = 1; i < lineElements.length; i++) {
					// normalize to bipolar vector
					double proportion = (Double.parseDouble(lineElements[i]) / 255);
					double toAdd = ((proportion * 2) - 1);
					testingData.add(toAdd);
				}

				y = new ArrayList<Double>();

				// Create copy of testing data
				for (int i = 0; i < image_dim; i++) {
					y.add(testingData.get(i));
				}

				converged = false;
				int epochs = 0;

				while (!converged) {
					converged = true;
					// Create random neuron testing order
					Collections.shuffle(list);
					// Calculating yin
					// k increments the value in the randomized neuron list
					// m increments the value in the matrix to calculate
					for (int i = 0; i < image_dim; i++) {
						double sum = 0;
						for (int m = 0; m < image_dim; m++) {
							sum += (y.get(m) * weightMatrix[m][list.get(i)]);
						}
						sum /= image_dim;
						yin = testingData.get(list.get(i)) + sum;
						yval = activationFunction(yin, steepness);
						if (Math.abs(y.get(list.get(i)) - yval) > errorThreshold) {
							y.set(list.get(i), yval);
							converged = false;
						}
					}
					epochs++;
				}
				// save results to file
				count++;
				saveResults(resultsFile, y, count, errorThreshold);
				System.out.println("Testing Pattern " + (index + 1)
						+ " converged after " + epochs + " epochs.");
				index++;
			}
		} catch (IOException e) {
			System.err.print("Error in I/O");
			System.exit(1);
		}

		System.out.println("Successfully tested. Results have been saved.");
	}

	/*
	 * Saves the weights of the trained Hopfield NN to a given weights file
	 * 
	 * @param weightMatrix the 2D array that represents weighted matrix
	 * 
	 * @param savedWeightsFile the name of the saved weights file
	 */
	public static void saveWeights(double[][] weightMatrix,
			String savedWeightsFile) {
		PrintWriter pw = null;

		try {
			pw = new PrintWriter(savedWeightsFile);
			for (int i = 0; i < weightMatrix.length; i++) {
				for (int j = 0; j < weightMatrix[i].length; j++) {
					pw.print(weightMatrix[i][j] + " ");
				}
				pw.println();
			}
		} catch (IOException e) {
			System.err.println("Error in I/O while saving weight matrix");
			System.exit(1);
		}

		pw.close();
	}

	/*
	 * Copies the weights from a saved file to a 2D array, representing the
	 * weight matrix
	 * 
	 * @param savedWeightsFile the name of the saved weights file
	 * 
	 * @param image_dim the dimension of the input image
	 * 
	 * @return int[][] a 2D array that represents the weight matrix
	 */
	public static double[][] weightMatrix(String savedWeightsFile, int image_dim) {
		// File I/O variables
		BufferedReader br = null;
		String toBeTokenized = null;
		StringTokenizer tokenizer = null;
		double[][] weightMatrix = new double[image_dim][image_dim];

		// read from saved weights file
		try {
			br = new BufferedReader(new FileReader(savedWeightsFile));
			toBeTokenized = br.readLine();
			tokenizer = new StringTokenizer(toBeTokenized);
			for (int i = 0; i < weightMatrix.length; i++) {
				for (int j = 0; j < weightMatrix[i].length; j++) {
					if (tokenizer.hasMoreTokens()) {
						weightMatrix[i][j] = Double.parseDouble(tokenizer
								.nextToken());
					} else {
						toBeTokenized = br.readLine();
						if (toBeTokenized == null)
							break;
						else {
							tokenizer = new StringTokenizer(toBeTokenized);
							weightMatrix[i][j] = Double.parseDouble(tokenizer
									.nextToken());
						}
					}
				}
			}
			br.close();
		} catch (IOException e) {
			System.err.print("Error in I/O");
			System.exit(1);
		}
		return weightMatrix;
	}

	/*
	 * Computes the activation of a given neuron
	 * 
	 * @param yin the computed dot product for the given neuron
	 * 
	 * @return int the activation value for the neuron
	 */
	public static double activationFunction(double yin, double steepness) {
		double denominator = 1 + Math.exp(-1 * steepness * yin);

		return (2 * (1 / denominator) - 1);
	}

	/*
	 * Displays the user menu
	 */
	public static void displayUserMenu() {
		System.out.println("\n");
		System.out.println("============================================");
		System.out.println("|                 USER MENU                 |");
		System.out.println("============================================");
		System.out.println("|       Operation:            Command:     |");
		System.out.println("|                                          |");
		System.out.println("| 1. Train the neural net.       1         |");
		System.out.println("| 2. Test the neural net.        2         |");
		System.out.println("| 3. Exit the program.           3         |");
		System.out.println("============================================");
	}

	/*
	 * Saves the results of the testing operation to an output file
	 * 
	 * @param resultsFile the name of the results file
	 * 
	 * @param y the converged results matrix
	 * 
	 * @param count the index of the testing set
	 */
	public static void saveResults(String resultsFile, ArrayList<Double> y,
			int count, double errorThreshold) {
		PrintWriter pw = null;
		BufferedReader br;
		String input;
		String[] lineElements = null;
		double[] results = new double[10];

		// clear file if first time opening it
		if (count == 1) {
			FileWriter fwOb;
			PrintWriter pwOb;
			try {
				fwOb = new FileWriter(resultsFile, false);
				pwOb = new PrintWriter(fwOb, false);
				pwOb.flush();
				pwOb.close();
				fwOb.close();
			} catch (FileNotFoundException e) {
				System.err.println(resultsFile + " not found");
				System.exit(1);
			} catch (IOException e) {
				System.err.println("Error in I/O");
				System.exit(1);
			}
		}

		double testValue;
		for (int i = 0; i < 10; i++) {
			try {
				br = new BufferedReader(new FileReader("matrix" + i + ".txt"));
				input = br.readLine();
				lineElements = input.split(" ");
				for (int j = 0; j < y.size(); j++) {
					testValue = Double.parseDouble(lineElements[j]);
					if (Math.abs(y.get(j) - testValue) < errorThreshold) {
						results[i]++;
					}
				}
				results[i] /= y.size();
			} catch (IOException e) {
				System.err.println("Error in I/O");
				System.exit(1);
			}
		}

		// write to output file
		int maxIndex = 0;
		double max = 0;
		DecimalFormat df;
		try {
			pw = new PrintWriter(new FileWriter(resultsFile, true));
			pw.write("Results from test sample #" + count + "\n");
			df = new DecimalFormat();
			df.setMaximumFractionDigits(2);
			for (int i = 0; i < results.length; i++) {
				pw.write("\tConfidence Interval of " + i + ": "
						+ df.format(results[i] * 100) + "%\n");
				if (results[i] > max) {
					max = results[i];
					maxIndex = i;
				}
			}
			pw.write("I am " + df.format(max * 100)
					+ "% sure this number is a " + maxIndex);
			pw.write("\n\n");
			pw.close();
		} catch (IOException e) {
			System.err.println("Error in I/O");
			System.exit(1);
		}
	}
	
	public static void statistics(String resultsFile) {
		BufferedReader br = null;
		String currentLine;
		String [] lineElements;
		
		int [] targets = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
		int count = 0;
		int target_index = 0;
		double correctlyClassified = 0;
		double totalClassified = 0;
		double [] percentageClassified = new double [targets.length];
		
		System.out.println("Computing statistics...");
		
		try{
			br = new BufferedReader(new FileReader(resultsFile));
			while((currentLine = br.readLine()) != null) {
				br.readLine();
				br.readLine();
				br.readLine();
				br.readLine();
				br.readLine();
				br.readLine();
				br.readLine();
				br.readLine();
				br.readLine();
				br.readLine();
				currentLine = br.readLine();
				lineElements = currentLine.split(" ");
				int classifiedNum = Integer.parseInt(lineElements[8]);
				if (classifiedNum == targets[target_index]) {
					correctlyClassified++;
				}
				totalClassified++;
				br.readLine();
				if(totalClassified == 50){
					percentageClassified[target_index] = (correctlyClassified / totalClassified);
					totalClassified = 0;
					correctlyClassified = 0;
					target_index++;
				}
			}
			
			for(int i = 0; i < percentageClassified.length; i++){
				System.out.println("Target: " + i);
				System.out.println("The Hopfield net classified "
						+ (percentageClassified[i] * 100)
						+ "% of the target patterns correctly.");
				System.out.println("------------------------");
			}
			
		} catch (IOException e) {
			System.err.println("Error in I/O");
			System.exit(1);
		}
	}
}