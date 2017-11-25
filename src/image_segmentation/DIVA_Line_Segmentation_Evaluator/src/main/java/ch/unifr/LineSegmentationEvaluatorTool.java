/*
 * Copyright (c) 2016 UniFR
 * University of Fribourg, Switzerland.
 */

package ch.unifr;

import org.apache.commons.cli.*;
import org.apache.log4j.Logger;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.Namespace;
import org.jdom2.input.SAXBuilder;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * LineSegmentationEvaluatorTool class of the ICDAR 2017 competition
 *
 * @author Manuel Bouillon <manuel.bouillon@unifr.ch>
 * @author Michele Alberti <michele.alberti@unifr.ch>
 * @date 25.07.2017
 * @brief The line segmentation evaluator main class
 */
public class LineSegmentationEvaluatorTool {

    /**
     * Log4j logger
     */
    private static final Logger logger = Logger.getLogger(LineSegmentationEvaluatorTool.class);

    /**
     * HisDoc Layout Competition Task-3(line segmentation) Evaluator
     */
    public static void main(String[] args) {
        logger.trace(Thread.currentThread().getStackTrace()[1].getMethodName());

        ///////////////////////////////////////////////////////////////////////////////////////////////
        // Parse parameters
        ///////////////////////////////////////////////////////////////////////////////////////////////
        Options options = new Options();

        // GT image
        Option igt = new Option("igt", "imageGroundTruth", true, "Ground Truth image at pixel-level");
        igt.setRequired(true);
        options.addOption(igt);

        // GT XML
        Option xgt = new Option("xgt", "xmlGroundTruth", true, "Ground Truth XML");
        xgt.setRequired(true);
        options.addOption(xgt);

        // Prediction XML
        Option xp = new Option("xp", "xmlPrediction", true, "Prediction XML");
        xp.setRequired(true);
        options.addOption(xp);

        // Output path, relative to prediction input path (optional)
        options.addOption(new Option("out", "outputPath", true, "Output path, relative to prediction input path"));

        // Overlap with original image (optional)
        options.addOption(new Option("overlap", true, "Overlap original image with the visualized results"));

        // Matching threshold (optional)
        options.addOption(new Option("mt", "matchingThreshold", true, "Matching threshold for detected lines"));

        // Save CSV file (optional)
        options.addOption(new Option("csv", false, "(Flag) Save the CSV file"));

        // Account for comments (optional)
        options.addOption(new Option("c", "comments", false, "(Flag) Take comments into account"));

        // Parse arguments
        CommandLine cmd;

        try {
            cmd = new DefaultParser().parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            new HelpFormatter().printHelp("utility-name", options);
            System.exit(1);
            return;
        }

        // Assign compulsory parameter values
        String imageGtPath = cmd.getOptionValue("imageGroundTruth").replace("/", File.separator);
        String xmlGtPath = cmd.getOptionValue("xmlGroundTruth").replace("/", File.separator);
        String xmlPredictionPath = cmd.getOptionValue("xmlPrediction").replace("/", File.separator);

        // Set the path of the prediction as starting output path
        String outputPath = xmlPredictionPath.substring(0,xmlPredictionPath.lastIndexOf(File.separator)+1);

        // Add any relative path from there (if specified)
        if(cmd.hasOption("outputPath")){
            outputPath += cmd.getOptionValue("outputPath").replace("/", File.separator);
        }

        // Make sure last char is a file separator
        if (outputPath.lastIndexOf(File.separator)+1 != outputPath.length()) {
            outputPath += File.separator;
        }

        // Add the prediction file name without extension
        outputPath += xmlPredictionPath.substring(xmlPredictionPath.lastIndexOf(File.separator) + 1, xmlPredictionPath.lastIndexOf('.'));

        double threshold = 0.75;
        if(cmd.hasOption("matchingThreshold")) {
            threshold = Double.parseDouble(cmd.getOptionValue("matchingThreshold"));
            logger.info("Matching threshold is: " + (100*threshold) + " %");
        }

        boolean comments = false;
        if(cmd.hasOption("comments")) {
            comments = true;
            logger.info("Taking comments into account: true");
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////
        // Parse input XMLs
        ///////////////////////////////////////////////////////////////////////////////////////////////

        // Loading the image GT
        BufferedImage image = null;
        logger.info("Loading image ground truth from " + imageGtPath);
        try {
            image = ImageIO.read(new File(imageGtPath));
        } catch (IOException e) {
            logger.error(e.getMessage());
        }

        // Loading GT XML
        logger.info("Loading page ground truth from " + xmlGtPath);
        // truth = ImageLinePageDataset.readDataFromFile(xmlGtPath, comments);
        List<Polygon> truth = ImageLinePageDataset.readDataFromFile(xmlGtPath);

        // Loading prediction XML
        logger.info("Loading method output from " + xmlPredictionPath);
        //output = ImageLinePageDataset.readDataFromFile(xmlPredictionPath, comments);
        List<Polygon> output = ImageLinePageDataset.readDataFromFile(xmlPredictionPath);

        // Evaluating the prediction provided
        logger.info("Evaluating...");
        LineSegmentationEvaluator evaluator = new LineSegmentationEvaluator();
        Results results = evaluator.evaluate(image, truth, output, threshold);

        // / Add the prediction filename to the results
        results.put(Results.FILENAME,xmlPredictionPath.substring(xmlPredictionPath.lastIndexOf(File.separator) + 1, xmlPredictionPath.lastIndexOf('.')));

        // Write the results in a CSV file, if outPath is provided
        if (cmd.hasOption("csv")) {
            logger.info("Writing results in " + outputPath);
            results.writeToCSV(xmlPredictionPath.substring(0, xmlPredictionPath.lastIndexOf(File.separator) + 1) + "results.csv");
        }

        // Write evaluation image
        BufferedImage visualization = evaluator.getEvalImage();
        try {
            ImageIO.write(visualization, "png", new File(outputPath+"-visualization.png"));
            logger.info("Writing visualization image in " + outputPath);
        } catch (IOException e) {
            logger.error(e);
        }

        // If desired, overlap the original image with the visualized result
        if(cmd.hasOption("overlap")){
            try {
                ImageIO.write(evaluator.overlapEvaluation(visualization, ImageIO.read(new File(cmd.getOptionValue("overlap")))), "png", new File(outputPath+"-overlap.png"));
                logger.info("Writing overlap image in " + outputPath);
            } catch (IOException e) {
                logger.error(e);
            }
        }
    }

    /**
     * Extract the main text area from a GT in XML format
     * @param xmlGtPath the GT file in XML format
     * @return a Rectangle representing the main text area
     */
    private static Rectangle getMainTextArea(String xmlGtPath) {
        String pointsString="";
        try {
            Element root = new SAXBuilder().build(new File(xmlGtPath)).getRootElement();
            Namespace namespace = root.getNamespace();
            Element page = root.getChild("Page", namespace);
            Element region = page.getChild("TextRegion", namespace);
            // Get the string with the coordinates from the XML
            pointsString = region.getChild("Coords", namespace).getAttributeValue("points");
        } catch (JDOMException | IOException e) {
            logger.error(e);
            if (logger.isDebugEnabled()) {
                e.printStackTrace();
            }
        }

        // Parse the points
        String[] pointsList = pointsString.split(" ");

        // Parse X and Y coordinates from the point list
        int[] x = {
                (int) Double.parseDouble(pointsList[0].split(",")[0]),
                (int) Double.parseDouble(pointsList[1].split(",")[0]),
                (int) Double.parseDouble(pointsList[2].split(",")[0]),
                (int) Double.parseDouble(pointsList[3].split(",")[0])};
        int[] y = {
                (int) Double.parseDouble(pointsList[0].split(",")[1]),
                (int) Double.parseDouble(pointsList[1].split(",")[1]),
                (int) Double.parseDouble(pointsList[2].split(",")[1]),
                (int) Double.parseDouble(pointsList[3].split(",")[1])};

        // Sort the coordinates to get easy max&min
        Arrays.sort(x);
        Arrays.sort(y);

        // Create the rectangle as: xMin, yMin, width=(xMax-xMin), height=(yMax - yMin)
        return new Rectangle(x[0], y[0], (x[x.length-1] - x[0]), (y[y.length-1] - y[0]));
    }
}


