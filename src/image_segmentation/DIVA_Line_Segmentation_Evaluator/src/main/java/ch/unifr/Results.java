/*
 * Copyright (c) 2016 UniFR
 * University of Fribourg, Switzerland.
 */

package ch.unifr;

import org.apache.log4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;

/**
 * Results class of the Experimenter project
 *
 * @author Manuel Bouillon <manuel.bouillon@unifr.ch>
 * @author Michele Alberti <michele.alberti@unifr.ch>
 * @date 25.07.2017
 * @brief Store results in a map.
 */
@SuppressWarnings({"WeakerAccess"})
public class Results {

    /**
     * Keys for the different measures
     */
    public static final String FILENAME = "LineSegmentation.filename.String";

    public static final String LINES_NB_TRUTH = "LineSegmentation.NbLinesTruth.int";
    public static final String LINES_NB_PROPOSED = "LineSegmentation.NbLinesProposed.int";
    public static final String LINES_NB_CORRECT = "LineSegmentation.NbLinesCorrect.int";

    public static final String LINES_IU = "LineSegmentation.LinesIU.double";
    public static final String LINES_FMEASURE = "LineSegmentation.LinesFMeasure.double";
    public static final String LINES_RECALL = "LineSegmentation.LinesRecall.Double";
    public static final String LINES_PRECISION = "LineSegmentation.LinesPrecision.Double";

    public static final String MATCHED_PIXEL_IU = "LineSegmentation.MatchedPixelIU.double";
    public static final String MATCHED_PIXEL_FMEASURE = "LineSegmentation.MatchedPixelFMeasure.Double";
    public static final String MATCHED_PIXEL_PRECISION = "LineSegmentation.MatchedPixelPrecision.Double";
    public static final String MATCHED_PIXEL_RECALL = "LineSegmentation.MatchedPixelRecall.Double";

    public static final String PIXEL_IU = "LineSegmentation.PixelIU.double";
    public static final String PIXEL_FMEASURE = "LineSegmentation.PixelFMeasure.Double";
    public static final String PIXEL_PRECISION = "LineSegmentation.PixelPrecision.Double";
    public static final String PIXEL_RECALL = "LineSegmentation.PixelRecall.Double";
    /**
     * Log4j logger
     */
    private static final Logger logger = Logger.getLogger(Results.class);
    /**
     * The map storing all the measures and their associated values
     */
    private Map<String, String> results = new HashMap<>();

    /**
     * Set/update the value associated with the key
     *
     * @param key   of the measure
     * @param value of the measure
     */
    public void put(String key, Object value) {
        logger.trace("put(" + key + ") = " + value);
        results.put(key, value.toString());
    }

    /**
     * Write results as CSV file. If the file already exists it appends a new line only
     * @param fName file name for the CSV results file
     */
    public void writeToCSV(String fName) {

        File file = new File(fName);

        // If the file does not exist, create it and write the name of the metrics
        if(!file.exists()){
            StringBuilder s = new StringBuilder();
            s.append(FILENAME.split("\\.")[1]).append(",");

            s.append(LINES_NB_TRUTH.split("\\.")[1]).append(",");
            s.append(LINES_NB_PROPOSED.split("\\.")[1]).append(",");
            s.append(LINES_NB_CORRECT.split("\\.")[1]).append(",");

            s.append(LINES_IU.split("\\.")[1]).append(",");
            s.append(LINES_FMEASURE.split("\\.")[1]).append(",");
            s.append(LINES_RECALL.split("\\.")[1]).append(",");
            s.append(LINES_PRECISION.split("\\.")[1]).append(",");

            s.append(MATCHED_PIXEL_IU.split("\\.")[1]).append(",");
            s.append(MATCHED_PIXEL_FMEASURE.split("\\.")[1]).append(",");
            s.append(MATCHED_PIXEL_PRECISION.split("\\.")[1]).append(",");
            s.append(MATCHED_PIXEL_RECALL.split("\\.")[1]).append(",");

            s.append(PIXEL_IU.split("\\.")[1]).append(",");
            s.append(PIXEL_FMEASURE.split("\\.")[1]).append(",");
            s.append(PIXEL_PRECISION.split("\\.")[1]).append(",");
            s.append(PIXEL_RECALL.split("\\.")[1]).append("\n");

            try {
                Files.write(Paths.get(fName), s.toString().getBytes());
                logger.debug("Created " + fName);
            } catch (IOException e) {
                logger.error(e.getMessage());
            }
        }

        StringBuilder s = new StringBuilder();
        s.append(results.get(FILENAME)).append(",");

        s.append(String.format("%2.4f", Double.parseDouble(results.get(LINES_NB_TRUTH)))).append(",");
        s.append(String.format("%2.4f", Double.parseDouble(results.get(LINES_NB_PROPOSED)))).append(",");
        s.append(String.format("%2.4f", Double.parseDouble(results.get(LINES_NB_CORRECT)))).append(",");

        s.append(String.format("%2.4f", Double.parseDouble(results.get(LINES_IU)))).append(",");
        s.append(String.format("%2.4f", Double.parseDouble(results.get(LINES_FMEASURE)))).append(",");
        s.append(String.format("%2.4f", Double.parseDouble(results.get(LINES_RECALL)))).append(",");
        s.append(String.format("%2.4f", Double.parseDouble(results.get(LINES_PRECISION)))).append(",");

        s.append(String.format("%2.4f", Double.parseDouble(results.get(MATCHED_PIXEL_IU)))).append(",");
        s.append(String.format("%2.4f", Double.parseDouble(results.get(MATCHED_PIXEL_FMEASURE)))).append(",");
        s.append(String.format("%2.4f", Double.parseDouble(results.get(MATCHED_PIXEL_PRECISION)))).append(",");
        s.append(String.format("%2.4f", Double.parseDouble(results.get(MATCHED_PIXEL_RECALL)))).append(",");

        s.append(String.format("%2.4f", Double.parseDouble(results.get(PIXEL_IU)))).append(",");
        s.append(String.format("%2.4f", Double.parseDouble(results.get(PIXEL_FMEASURE)))).append(",");
        s.append(String.format("%2.4f", Double.parseDouble(results.get(PIXEL_PRECISION)))).append(",");
        s.append(String.format("%2.4f", Double.parseDouble(results.get(PIXEL_RECALL)))).append("\n");

        try {
            Files.write(Paths.get(fName), s.toString().getBytes(), StandardOpenOption.APPEND);
            logger.debug("Appended results on" + fName);
        } catch (IOException e) {
            logger.error(e.getMessage());
        }

    }
}
