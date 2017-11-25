/*
 * Copyright (c) 2016 UniFR
 * University of Fribourg, Switzerland.
 */

package ch.unifr;

import javafx.util.Pair;
import org.apache.log4j.Logger;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;

/**
 * LineSegmentationEvaluator class of the Experimenter project
 *
 * @author Manuel Bouillon <manuel.bouillon@unifr.ch>
 * @author Michele Alberti <michele.alberti@unifr.ch>
 * @date 25.07.2017
 * @brief Line segmentation evaluator
 * Find all possible matching between Grount Truth (GT) polygons and
 * Method Output (MO) polygons and compute the matching possibilities areas.
 * Final matching is done starting from the biggest matching possibility area
 * and continuing until all polygons are match or until no matching possibility remains.
 */
@SuppressWarnings({"WeakerAccess"})
public class LineSegmentationEvaluator {

    /**
     * Log4j logger
     */
    protected static final Logger logger = Logger.getLogger(LineSegmentationEvaluator.class);
    /**
     * Evaluation image
     */
    private BufferedImage evalImage = null;

    /**
     * Evaluate output data with respect to ground truth
     *
     * @param groundTruthImage         the ground truth groundTruthImage
     * @param prediction  the polygons output by the method to evaluate
     * @param groundTruth   the ground truth polygons
     * @param threshold     the IU threshold for line matching
     * @return Results object
     */
    public Results evaluate(BufferedImage groundTruthImage, List<Polygon> groundTruth, List<Polygon> prediction, double threshold) {
        logger.trace(Thread.currentThread().getStackTrace()[1].getMethodName());

        // Match overlapping polygons
        List<Pair<Polygon,Polygon>> matching = getMatchingPolygons(groundTruthImage, groundTruth, prediction);

        // Init evaluation image
        evalImage = new BufferedImage(groundTruthImage.getWidth(), groundTruthImage.getHeight(), BufferedImage.TYPE_INT_RGB);

        // Lines count
        int nbLinesCorrect = 0;
        int nbLinesMissed = 0;
        int nbLinesExtra = 0;

        // Pixels counts for the matched lines
        int matchedTP = 0;
        int matchedFN = 0;
        int matchedFP = 0;

        // Pixels counts for the whole image
        int TP = 0;
        int FN = 0;
        int FP = 0;
        int nbPixelsPrediction = 0;
        int nbPixelsGt = 0;

        // For every match
        for (Pair<Polygon,Polygon> match : matching) {

            // Extract the predicted and ground truth polygons from the match pair
            Polygon pp = match.getKey();
            Polygon pgt = match.getValue();

            logger.trace("evaluation matching " + pgt + " * " + pp);

            // Pixels counts for the line (current match of polygons)
            int lineTP = 0; // True positive pixels
            int lineFN = 0; // False negative pixels
            int lineFP = 0; // False positive pixels
            int lineNbPixelsPrediction = 0;
            int lineNbPixelsGt = 0;

            // These lines are for deep MANUAL inspection only (especially for the visualization!)
            //if(pp!=null && pgt!=null)continue; // Skip all correctly matched lines
            //if(pp==null || pgt==null)continue; // Skip all the extra and missed lines

            /* Find the bounding box of both polygons, i.e the bounding box of the union
             * In case one of the two polygons is null (because it was an extra o miss line)
             * the union is exactly the non-null polygon.
             */
            Rectangle rp = (pp!=null) ? pp.getBounds(): pgt.getBounds();
            Rectangle rgt = (pgt!=null) ? pgt.getBounds(): pp.getBounds();

            // Find the union
            Rectangle union = rgt.union(rp);

            // For every pixel in the bounding box
            for (int x = (int) union.getMinX(); x < union.getMaxX(); x++) {
                for (int y = (int) union.getMinY(); y < union.getMaxY(); y++) {

                    // Ignore boundary pixels
                    if (((groundTruthImage.getRGB(x,y) >> 23) & 0x1) == 1) {
                        continue;
                    }

                    // Ignore background pixels
                    if ((groundTruthImage.getRGB(x, y) & 0x1) == 1) {
                        continue;
                    }

                    // Check the type of pixel: TP, FN, FP (it cannot be a TN here, we're iterating on the union.
                    boolean isInPp = (pp != null) && pp.contains(x, y);
                    boolean isInPgt = (pgt != null) && pgt.contains(x, y);

                    if (isInPp && isInPgt) {           // Predicted correctly
                        lineTP++;
                    } else if (!isInPp && isInPgt) {   // Not predicted (but it should have been)
                        lineFN++;
                    } else if (isInPp && !isInPgt) {   // Predicted (but it should NOT have been)
                        lineFP++;
                    }

                    // Update pmo size
                    if (isInPp) {
                        lineNbPixelsPrediction++;
                    }

                    // Update pgt size
                    if (isInPgt) {
                        lineNbPixelsGt++;
                    }

                    // Update visualization image
                    /*
                     * (0x007F00) GREEN:   Foreground predicted correctly
                     * (0xFFFF00) YELLOW:  Foreground which belong two multiple lines (all cases)
                     * (0xFF0000) RED:     Foreground does not belong to this line (False positive)
                     * (0x00FFFF) BLUE:    Foreground that should have been in this (False negative)
                     */
                    // Draw only if it concerns this line
                    if(isInPgt || isInPp) {
                        int color = 0x0;                   // Black
                        if (isInPp && isInPgt) {
                            color = 0x007F00;              // Green
                        } else if (isInPp && !isInPgt) {
                            color = 0xFF0000;              // Red
                        } else if (!isInPp && isInPgt) {
                            color = 0x0088FF;              // Blue
                        }

                        // Get the current color of the visualization
                        int current = evalImage.getRGB(x, y) & 0x00FFFFFF;
                        // If its not black and its not the same with want to apply -> it must be yellow!
                        if (current != 0 && current != color) {
                            evalImage.setRGB(x, y, 0xFFFF00);    // Yellow
                        } else {
                            evalImage.setRGB(x, y, color);
                        }
                    }

                }
            }

            // Drawing polygon color
            Color color = Color.WHITE;

            // Integrate values for this line into the global sum
            TP += lineTP;
            FN += lineFN;
            FP += lineFP;
            nbPixelsPrediction += lineNbPixelsPrediction;
            nbPixelsGt += lineNbPixelsGt;

            // Evaluate the line detection
            // NOTE: a line can be considered as both miss and extra if the conditions are met!
            double P = lineTP / (double) (lineTP+lineFP); // Precision
            double R = lineTP / (double) (lineTP+lineFN); // Recall
            logger.trace("P = " + P);
            logger.trace("R = " + R);

            ///////////////////////////////////////////////////////////////////////////////////////
            // The line hit too many extra pixels which did not belong to the GT, hence is considered an extra line
            if(P < threshold) {
                logger.debug("line considered as extra");
                nbLinesExtra++;
                color = Color.RED;
            }

            ///////////////////////////////////////////////////////////////////////////////////////
            // The line hit too few pixels which belong to the GT, hence is considered as a miss line
            if(R < threshold) {
                logger.debug("line considered as  missed");
                nbLinesMissed++;
                color = Color.BLUE;
            }

            ///////////////////////////////////////////////////////////////////////////////////////
            // The line is considered as correctly detected
            if(P >= threshold && R >= threshold) {
                logger.trace("line considered as correctly detected");
                // Integrate values for this line into the global sum
                matchedTP += lineTP;
                matchedFN += lineFN;
                matchedFP += lineFP;
                nbLinesCorrect++;
                color = Color.GREEN;
            } else {
                logger.debug("line skipped, P|R below threshold: P=" + P + ",R=" + R);
            }

            // For coloring the polygon in case the line has both too low R and P
            if(P < threshold && R < threshold) {
                color = Color.PINK;
            }

            ///////////////////////////////////////////////////////////////////////////////////////
            // Draw the polygon on the visualization
            if(pp!=null){
                Graphics g = evalImage.getGraphics();
                g.setColor(color);
                g.drawPolygon(pp);
            }
        }

        // Line scores
        double linePrecision = nbLinesCorrect / (double) (nbLinesCorrect + nbLinesExtra);
        double lineRecall = nbLinesCorrect / (double) (nbLinesCorrect + nbLinesMissed);
        double lineF1 = 2 * nbLinesCorrect / (double) (2 * nbLinesCorrect + nbLinesMissed + nbLinesExtra);
        double lineIU = nbLinesCorrect / (double) (nbLinesCorrect + nbLinesMissed + nbLinesExtra);

        // Pixel scores
        double matchedPixelPrecision = matchedTP / (double) (matchedTP + matchedFP);
        double matchedPixelRecall = matchedTP / (double) (matchedTP + matchedFN);
        double matchedPixelF1 = 2 * matchedTP / (double) (2 * matchedTP + matchedFP + matchedFN);
        double matchedPixelIU = matchedTP / (double) (matchedTP + matchedFP + matchedFN);

        // Pixel scores
        double pixelPrecision = TP / (double) (TP + FP);
        double pixelRecall = TP / (double) (TP + FN);
        double pixelF1 = 2 * TP / (double) (2 * TP + FP + FN);
        double pixelIU = TP / (double) (TP + FP + FN);

        // Storing line results
        Results results = new Results();

        results.put(Results.LINES_NB_TRUTH, groundTruth.size());
        results.put(Results.LINES_NB_PROPOSED, prediction.size());
        results.put(Results.LINES_NB_CORRECT, nbLinesCorrect);

        results.put(Results.LINES_IU, lineIU);
        results.put(Results.LINES_FMEASURE, lineF1);
        results.put(Results.LINES_RECALL, lineRecall);
        results.put(Results.LINES_PRECISION, linePrecision);

        // Storing matched pixel results
        results.put(Results.MATCHED_PIXEL_IU, matchedPixelIU);
        results.put(Results.MATCHED_PIXEL_FMEASURE, matchedPixelF1);
        results.put(Results.MATCHED_PIXEL_PRECISION, matchedPixelPrecision);
        results.put(Results.MATCHED_PIXEL_RECALL, matchedPixelRecall);

        // Storing pixel results
        results.put(Results.PIXEL_IU, pixelIU);
        results.put(Results.PIXEL_FMEASURE, pixelF1);
        results.put(Results.PIXEL_PRECISION, pixelPrecision);
        results.put(Results.PIXEL_RECALL, pixelRecall);

        logger.trace(results.getClass().getName() + "@" + Integer.toHexString(System.identityHashCode(results)));

        // Logging
        logger.debug("TP = " + TP);
        logger.debug("FP = " + FP);
        logger.debug("FN = " + FN);
        logger.debug("GT size = " + groundTruth.size());

        logger.debug("Prediction size = " + prediction.size());
        logger.debug("nbPixelsPrediction = " + nbPixelsPrediction);
        logger.debug("nbPixelsGt = " + nbPixelsGt);
        logger.debug("nbLinesCorrect = " + nbLinesCorrect);
        logger.debug("nbLinesExtra = " + nbLinesExtra);
        logger.debug("nbLinesMissed = " + nbLinesMissed);

        logger.debug("line IU = " + lineIU);
        logger.debug("line F1 = " + lineF1);
        logger.debug("linePrecision = " + linePrecision);
        logger.debug("lineRecall = " + lineRecall);

        logger.debug("matchedPixel IU = " + matchedPixelIU);
        logger.debug("matchedPixel F1 = " + matchedPixelF1);
        logger.debug("matchedPixelPrecision = " + matchedPixelPrecision);
        logger.debug("matchedPixelRecall = " + matchedPixelRecall);

        logger.debug("pixel IU = " + pixelIU);
        logger.debug("pixel F1 = " + pixelF1);
        logger.debug("pixelPrecision = " + pixelPrecision);
        logger.debug("pixelRecall = " + pixelRecall);

        return results;
    }

    /**
     * Find the best matching polygons between the prediction and the groundTruth
     *
     * @param prediction polygons given by the method
     * @param groundTruth  polygons in the ground truth
     * @return the matching polygons
     */
    private List<Pair<Polygon,Polygon>> getMatchingPolygons(BufferedImage groundTruthImage, List<Polygon> groundTruth, List<Polygon> prediction) {
        logger.trace(Thread.currentThread().getStackTrace()[1].getMethodName());

        // Init the return value (the match)
        List<Pair<Polygon,Polygon>> matching = new ArrayList<>();
        Set<Polygon> matchedPolygons = new HashSet<>();

        // Init the list of all possibilities
        ArrayList<Possibility> possibilities = new ArrayList<>();

        /* Measure the score between each pair of polygons \in GT U P,
         * where GT and P represent the set of polygons for the GT and the
         * prediction respectively.
         * The outcome is a list of maximal size |GT|X|P| where each element of
         * the list is a triplet (Possibility.class) which stores a possible
         * match between two polygons and their score (in this case the UI).
         * Triplets with the trivial score 0 (no overlap between the bounds of
         * the polygons) are omitted in the list.
         */
        // For every GT polygon
        for (Polygon pgt : groundTruth) {

            // Find bounding box of GT
            Rectangle rgt = pgt.getBounds();
            logger.trace("matching possibility for GT: " + pgt);

            // For every Prediction polygon
            for (Polygon pp : prediction) {

                // Find bounding box of prediction
                Rectangle rp = pp.getBounds();

                // Skip if no overlap
                if (!rgt.intersects(rp)) {
                    logger.trace("no matching possibility of " + pgt + " with MO: " + pp);
                    continue;
                }

                // Find the union
                Rectangle union = rgt.union(rp);

                // Iterate the union area looking for foreground pixels belonging to both polygons
                int intersectingPixels = 0;
                int unionPixels = 0;
                for (int x = (int) union.getMinX(); x < union.getMaxX(); x++) {
                    for (int y = (int) union.getMinY(); y < union.getMaxY(); y++) {
                        // Ignore boundary pixels
                        if (((groundTruthImage.getRGB(x,y) >> 23) & 0x1) == 1) {
                            continue;
                        }

                        // Ignore background pixels
                        if ((groundTruthImage.getRGB(x, y) & 0x1) == 1) {
                            continue;
                        }

                        // Check if pixels belong to polygons
                        boolean isInPp = pp.contains(x, y);
                        boolean isInPgt = pgt.contains(x, y);

                        // If the pixel belongs to both the polygons
                        if (isInPp && isInPgt) {
                            intersectingPixels++;
                        }

                        // If the pixel belongs any of the polygons
                        if (isInPp || isInPgt) {
                            unionPixels++;
                        }
                    }
                }

                // Omit trivial '0' results
                if(intersectingPixels > 0) {
                    // Add the matching possibility
                    possibilities.add(new Possibility(pgt, pp, intersectingPixels/(double)unionPixels));
                    logger.trace("matching possibility: " + pgt + " * " + pp + " = " + intersectingPixels/(double)unionPixels);
                }
            }
        }
        logger.debug(possibilities.size() + " possibilities");

        /* Traverse the score-descending sorted list of Possibility and select
         * the first available match for each polygon belonging to the Prediction set.
         * This ensures that no polygons are matched twice and that each polygon
         * belonging to P gets matched with is best (available) matching polygon in
         * the GT, thus maximizing the total matching score in a deterministic way.
         */
        Collections.sort(possibilities);
        for (Possibility p : possibilities){
            // Take the next one free on the sorted list
            if (!matchedPolygons.contains(p.p) && !matchedPolygons.contains(p.gt)){
                // Add matching polygons
                logger.debug("match " + p.score);
                matching.add(new Pair<>(p.p,p.gt));
                matchedPolygons.add(p.p);
                matchedPolygons.add(p.gt);
            }
        }

        logger.info("found " + matching.size() + " matches");

        // Add all missing GT polygons (un-matched) by matching them will 'null'
        for (Polygon pgt : groundTruth) {
            if (!matchedPolygons.contains(pgt)){
                logger.debug("missed line matched with null");
                matching.add(new Pair<>(null,pgt));
                matchedPolygons.add(pgt);
            }
        }
        // Add all extra Prediction polygons (un-matched) by matching them will 'null'
        for (Polygon pp : prediction) {
            if (!matchedPolygons.contains(pp)){
                logger.debug("extra line matched with null");
                matching.add(new Pair<>(pp,null));
                matchedPolygons.add(pp);
            }
        }

        // Check that all polygons got eventually matched
        if(matchedPolygons.size() != groundTruth.size() + prediction.size()){
            logger.error("ERROR: some polygons have not been matched!");
        }

        logger.trace(matching.getClass().getName() + "@" + Integer.toHexString(System.identityHashCode(matching)));

        return matching;
    }

    /**
     * Get the evaluation image
     * @return eval image
     */
    public BufferedImage getEvalImage() {
        return evalImage;
    }

    /**
     * This method overlaps the evaluation visualization with the original image to further
     * enable the user to spot and interpret the mistakes in the prediction
     *
     * @param visualization the visualization image generated by this.getEvalImage()
     * @param original      the original image as it is in the dataset
     * @return a BufferedImage representing the overlapped images
     */
    public BufferedImage overlapEvaluation(BufferedImage visualization, BufferedImage original) {

        assert (visualization.getWidth() == original.getWidth());
        assert (visualization.getHeight() == original.getHeight());

        // Create new image of type ARGB (with alpha channel)
        BufferedImage overlap = new BufferedImage(visualization.getWidth(), visualization.getHeight(), BufferedImage.TYPE_INT_ARGB);

        Graphics g = overlap.getGraphics();

        // Paint original
        g.drawImage(original, 0, 0, null);

        // Set alpha
        ((Graphics2D) g).setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.57f));

        // Paint visualization
        g.drawImage(visualization, 0, 0, null);

        return overlap;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // PRIVATE
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * This class represent a triplet of two polygons (GT and prediction) and their
     * matching score (typically the IU). It is used to represent a possible match between
     * a GT polygon and a Prediction one.
     */
    private final class Possibility implements Comparable<Possibility>{
        /**
         * The polygon belonging to the GT
         */
        public final Polygon gt;
        /**
         * The polygon belonging to the prediction
         */
        public final Polygon p;
        /**
         * The matching score between the two (typically the IU of their bounds)
         */
        public final double score;

        /**
         * Build a Possibility (triplet)
         * @param gt the gt polygon
         * @param p the prediction polygon
         * @param score their matching score
         */
        public Possibility(Polygon gt, Polygon p, double score) {
            this.gt = gt;
            this.p = p;
            this.score = score;
        }

        /**
         * Comparator on the score of possibilities
         * @param p the possibility to comapre to
         * @return standard Double.compare (-1,0,1)
         */
        @Override
        public int compareTo(Possibility p) {
            return Double.compare(p.score,score);
        }
    }
}
