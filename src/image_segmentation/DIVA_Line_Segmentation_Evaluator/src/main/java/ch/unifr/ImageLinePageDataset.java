/*
 * Copyright (c) 2016 UniFR
 * University of Fribourg, Switzerland.
 */

package ch.unifr;

import org.apache.log4j.Logger;
import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.Namespace;
import org.jdom2.input.SAXBuilder;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * ImageLinePageDataset class of the Experimenter project
 *
 * @author Manuel Bouillon <manuel.bouillon@unifr.ch>
 * @review Michele Alberti <michele.alberti@unifr.ch>
 * @date 25.07.2017
 * @brief Load data files in PAGE format
 */
public class ImageLinePageDataset {

    /**
     * Log4j logger
     */
    private static final Logger logger = Logger.getLogger(ImageLinePageDataset.class);
    /**
     * TASKTAG should be "Coords" for line segmentation and "Baseline" for baseline extraction
     */
    private static final String TASKTAG = "Coords";

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC STATIC
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Load output data
     *
     * @param path the output data path
     * @return the output data object
     */
    public static List<Polygon> readDataFromFile(final String path) {
        logger.trace(Thread.currentThread().getStackTrace()[1].getMethodName());

        // Getting XML doc from PAGE ground truth
        Document xmlDocument = null;
        try {
            xmlDocument = new SAXBuilder().build(new File(path));
        } catch (JDOMException | IOException e) {
            logger.error("cannot open file: " + path);
            if (logger.isDebugEnabled()) {
                e.printStackTrace();
            }
        }

        List<Polygon> lines = getPolygonFromXml(xmlDocument);

        String classname = (lines == null) ? "null" : lines.getClass().getName();
        logger.trace(classname + "@" + Integer.toHexString(System.identityHashCode(lines)));
        return lines;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // PRIVATE STATIC
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Convert an XML document into a list of polygons
     *
     * @param xmlDocument the XML ground truth
     * @return a list of polygon
     */
    private static List<Polygon> getPolygonFromXml(Document xmlDocument) {
        logger.trace(Thread.currentThread().getStackTrace()[1].getMethodName());
        if (xmlDocument == null) {
            logger.error("cannot extract polygons from null xml document");
            return null;
        }

        List<Polygon> polygons = new ArrayList<>();

        Element root = xmlDocument.getRootElement();
        Namespace namespace = root.getNamespace();
        Element page = root.getChild("Page", namespace);
        List<Element> textRegions = page.getChildren("TextRegion", namespace);

        // Parsing structure of standard Page Xml…
        for (Element region : textRegions) {

            // Find the text region corresponding to main text area
            if (region.getAttribute("id").getValue().equals("region_textline")) {

                // Get all the text lines
                List<Element> textLines = region.getChildren("TextLine", namespace);
                for (Element line : textLines) {

                    Polygon polygon = new Polygon();

                    // Get the list of point and split it
                    String coordString = line.getChild(TASKTAG, namespace).getAttributeValue("points");
                    String[] coords = coordString.split(" ");

                    // For each point
                    for (int j = 0; j < coords.length; j++) {

                        // Split x and y
                        String[] c = coords[j].split(",");
                        int x = Integer.parseInt(c[0]);
                        int y = Integer.parseInt(c[1]);

                        // Add point
                        polygon.addPoint(x, y);
                    }

                    // add the polygon
                    polygons.add(polygon);
                }
            }
        }

        logger.debug("found " + polygons.size() + " polygons in XML document");
        return polygons;
    }
}

