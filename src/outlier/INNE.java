package outlier;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class INNE {


    private class NNSet {
        private int[] idxPoints;
        private double[] enclosureRadius;
        private double[] score;

        public NNSet(int size) {
            this.idxPoints = new int[size];
            this.enclosureRadius = new double[size];
            this.score = new double[size];
        }

    }

    private int numSub;
    private int numSet;
    private int numAttributes;
    private int numInstances;
    private double elapsedTrainingTime;
    private double elapsedEvaluationTime;
    private List<NNSet> ensemble = null;
    private double[] scores = null;
    private Instances instances;
    private double auc;
    private Random random = null;
    private boolean hasLabels;

    String dataFileName;
    String datasetName;

    public INNE(String[] args) {
        try {
            initVariables(args);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public INNE(int numSub, int numSet, Instances instances) {
        this.numSet = numSet;
        this.instances = instances;
        this.dataFileName = instances.relationName();

        numAttributes = instances.numAttributes();
        numInstances = instances.numInstances();

        this.numSub = Math.min(numSub, numInstances);
        random = new Random();
        ensemble = new ArrayList<NNSet>();
    }

    public void run() throws Exception {
        final long time1 = System.nanoTime();
        createEnsemble();
        final long time2 = System.nanoTime();
        elapsedTrainingTime = (time2 - time1) / 1000000000.0;
        System.out.println("iNNE models created. Training time: " + elapsedTrainingTime + " seconds.");

        calculateScores();
        final long time3 = System.nanoTime();
        elapsedEvaluationTime = (time3 - time2) / 1000000000.0;
        System.out.println("Anomaly scores calculated. Evaluation time: " + elapsedEvaluationTime + " seconds.");

        if(hasLabels) calAUC();
    }


    public void writeOutput() throws IOException {

        if(hasLabels) {
            File fileAUCOut = new File("./AUC_iNNE_Dataset_" + datasetName + ".csv");
            boolean addHeader = false;
            if(!fileAUCOut.exists()) addHeader = true;

            PrintWriter printWriter = new PrintWriter(new FileWriter(fileAUCOut, true));
            if(addHeader) printWriter.println("Ensemble Size" + "," + "Sample Size" + "," + "AUC" + "," + "Training Time" + "," + "Evaluation Time");
            printWriter.println(this.numSet + "," + this.numSub + "," + this.auc + "," + this.elapsedTrainingTime + "," + this.elapsedEvaluationTime);
            printWriter.close();
        }

		File fileOutput = new File("./Scores_Dataset_" + datasetName + ".csv");
		PrintWriter fOut = new PrintWriter(new FileWriter(fileOutput));

		fOut.println("Id,Label,Anomaly Score");

		for (int i = 0; i < this.numInstances; i++)
		{
			fOut.println(i + "," + instances.instance(i).value(numAttributes) + "," + this.scores[i]);
		}

		fOut.close();

    }

    private void calAUC() {
        long tp = 0L;
        long fp = 0L;
        auc = 0.0D;

        int[] idx = Utils.stableSort(this.scores);

        for (int i = 0; i < this.numInstances; i++) {
            if (instances.instance(idx[i]).value(numAttributes) == 1.0D) {
                tp += 1L;
            } else {
                auc += tp;
                fp += 1L;
            }
        }

        auc /= (tp * fp);
        System.out.println("AUC : " + auc);
    }

    private void calculateScores() {
        scores = new double[numInstances];

        for (int i = 0; i < numInstances; i++) {
            double[] instance = instances.instance(i).toDoubleArray();
            scores[i] = 0.0D;

            for (int j = 0; j < numSet; j++) {
                NNSet nnset = ensemble.get(j);
                double minRadius = Double.MAX_VALUE;
                double score = 1.0D;

                for (int k = 0; k < numSub; k++) {
                    double distance = calcDistance(instance, instances.instance(nnset.idxPoints[k]).toDoubleArray());
                    if (distance <= nnset.enclosureRadius[k] && nnset.enclosureRadius[k] < minRadius) {
                        minRadius = nnset.enclosureRadius[k];
                        score = nnset.score[k];
                    }
                }

                scores[i] += score;

            }
        }

        for (int i = 0; i < numInstances; i++) {
            scores[i] /= numSet;
        }

    }

    private int[] getRandomPermutation(int length, Random r) {

        // initialize array and fill it with {0,1,2...}
        int[] array = new int[length];
        for (int i = 0; i < array.length; i++)
            array[i] = i;

        for (int i = 0; i < length; i++) {

            int ran = i + r.nextInt(length - i);

            // perform swap
            int temp = array[i];
            array[i] = array[ran];
            array[ran] = temp;
        }
        return array;
    }

    private void createEnsemble() {
        int[] randomPermutation = getRandomPermutation(numInstances, random);
        double[][] pDist = new double[numSub][numSub];
        int[] minIdx = new int[numSub];
        int currentIndex = 0;

        for (int i = 0; i < numSet; i++) {
            NNSet nnset = new NNSet(numSub);
            if ((currentIndex + numSub) > numInstances) {
                randomPermutation = getRandomPermutation(numInstances, random);
                currentIndex = 0;
            }
            System.arraycopy(randomPermutation, currentIndex, nnset.idxPoints, 0, numSub);
            currentIndex += numSub;

            for (int n = 0; n < numSub; n++) {
                for (int m = 0; m < numSub; m++) {
                    if (m == n) {
                        pDist[m][m] = 0;
                    } else {
                        pDist[m][n] = pDist[n][m] = calcDistance(instances.instance(nnset.idxPoints[m]).toDoubleArray(), instances.instance(nnset.idxPoints[n]).toDoubleArray());
                    }
                }
            }

            for (int n = 0; n < numSub; n++) {
                minIdx[n] = -1;
                double minRad = Double.MAX_VALUE;
                ;
                for (int idx = 0; idx < numSub; idx++) {
                    if (n != idx && minRad > pDist[n][idx]) {
                        minRad = pDist[n][idx];
                        minIdx[n] = idx;
                    }
                }
                nnset.enclosureRadius[n] = minRad;
            }

            for (int n = 0; n < numSub; n++) {
                if (nnset.enclosureRadius[n] == 0) {
                    nnset.score[n] = 0.0D;
                } else {
                    nnset.score[n] = 1.0D - (nnset.enclosureRadius[minIdx[n]] / nnset.enclosureRadius[n]);
                }
            }

            ensemble.add(nnset);

        }

    }

    private double calcDistance(double[] d1, double[] d2) {
        double distance = 0.0;

        for (int i = 0; i < numAttributes; i++) {
            distance += (d1[i] - d2[i]) * (d1[i] - d2[i]);
        }

        if (distance > 0.0D) distance = Math.sqrt(distance);
        return distance;
    }


    private void initVariables(String[] args) throws Exception {
        String strNumSub = Utils.getOption('S', args);
        String strNumSet = Utils.getOption('T', args);

        if (strNumSub.length() != 0) {
            this.numSub = Integer.parseInt(strNumSub);
        } else {
            this.numSub = 8;
        }

        if (strNumSet.length() != 0) {
            this.numSet = Integer.parseInt(strNumSet);
        } else {
            this.numSet = 100;
        }

        dataFileName = Utils.getOption("dataFile", args);
        File temp = new File(dataFileName);
        if (!temp.exists()) throw new FileNotFoundException();

        if (Utils.getFlag("hasLabels", args)) hasLabels = true;

        String fileType = Utils.getOption("fileType", args);

        if (fileType.isEmpty()) fileType = "arff";

        if (fileType.equalsIgnoreCase("arff")) {
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataFileName);
            instances = source.getDataSet();
            datasetName = dataFileName.substring(0,dataFileName.length()-6);
        } else if (fileType.equalsIgnoreCase("csv")) {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(dataFileName));
            instances = loader.getDataSet();
            datasetName = dataFileName.substring(0,dataFileName.length()-5);
        }

        numAttributes = instances.numAttributes();
        numInstances = instances.numInstances();
        if (hasLabels) numAttributes = numAttributes - 1;

        numSub = Math.min(numSub, numInstances);
        random = new Random();
        ensemble = new ArrayList<NNSet>();
    }


    public static void main(String[] args) {
        try {
            INNE iNNE = new INNE(args);
            iNNE.run();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
