import datetime
import re
import xml.dom.minidom as minidom
import xml.etree.cElementTree as ET

import numpy as np


def writePAGEfile(outpath, textLines="", textRegionCoords="not provided", baselines=None):
    # Create root element and add the attributes
    root = ET.Element("PcGts")
    root.set("xmls", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:schemaLocation", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd")

    # Add metadata
    metadata = ET.SubElement(root, "Metadata")
    ET.SubElement(metadata, "Creator").text = "Michele Alberti, Vinaychandran Pondenkandath"
    ET.SubElement(metadata, "Created").text = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    ET.SubElement(metadata, "LastChange").text = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')

    # Add page
    page = ET.SubElement(root, "Page")

    # Add TextRegion
    textRegion = ET.SubElement(page, "TextRegion")
    textRegion.set("id", "region_textline")
    textRegion.set("custom", "0")

    # Add Coords
    ET.SubElement(textRegion, "Coords", points=textRegionCoords)

    # Add TextLine
    for i, line in enumerate(textLines):
        textLine = ET.SubElement(textRegion, "TextLine", id="textline_{}".format(i), custom="0")
        ET.SubElement(textLine, "Coords", points=line)
        if baselines:
            ET.SubElement(textLine, "Baseline", points=baselines[i])
        else:
            ET.SubElement(textLine, "Baseline", points="not provided")
        textEquiv = ET.SubElement(textLine, "TextEquiv")
        ET.SubElement(textEquiv, "Unicode")

    # Add TextEquiv to textRegion
    textEquiv = ET.SubElement(textRegion, "TextEquiv")
    ET.SubElement(textEquiv, "Unicode")

    #print(prettify(root))

    # Save on file
    file = open(outpath, "w")
    file.write(prettify(root))
    file.close()


def read_max_textline_from_file(pageFile):
    tree = ET.parse(pageFile)
    root = tree.getroot()
    NSMAP = {'pr': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    id = 0
    for textregion in root[1].findall('.//pr:TextRegion', namespaces=NSMAP):
        if 'textline' in textregion.attrib['id']:
            for textline in textregion.findall('.//pr:TextLine', namespaces=NSMAP):
                str = textline.attrib['id']
                id = np.max([id, int(re.findall(r'\d+', str)[0])])
    return id+1


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")
