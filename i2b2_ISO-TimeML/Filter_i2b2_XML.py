# Filters an XML file to keep only the timex expressions that overlap by at least one position in gold.
# Amy Olex
# 1/9/22

import os
from xml.etree.ElementTree import tostring
#import utils
import argparse

def get_timex_list(xml_path):
    """
    Function to read in an i2b2 XML file and extract out the TIMEX list of annotations.

    :param xml_path: Path to all the XML files.
    :return: Returns the text and the list of timex tags.
    """
    # Parsing XML Tags
    try:
        xml_parsed = parse(xml_path)
    except:
        print(xml_path)
        raise

    text_containers = xml_parsed.findall('TEXT')
    #print("My TEXT: " + str(text_containers[0].text))

    tag_containers = xml_parsed.findall('TAGS')
    assert len(tag_containers) == 1, "Found multiple tag sets!"
    tag_container = tag_containers[0]

    timex_tags = tag_container.findall('TIMEX3')

    return text_containers[0].text, timex_tags

def intersect_timex(timex1, timex2, chrono=False):
    """
    Takes in 2 timex xmltree lists and returns the intersection based on coordinates.
    Timex1 is the list that will be filtered and returned.
    :param timex1: The timex list with coords that will be returned
    :param timex2: The timex list with coords that will be used for filtering
    :param chrono: a boolean indicating if the timex1 coords are from chrono, if yes then add 1 to all.
    :return: a list of timexes from timex1 that intersected based on span with timex2
    """

    intersect = []

    for t1 in timex1:
        t1_label = t1.attrib['type']
        t1_start_pos, t1_end_pos = int(t1.attrib['start']), int(t1.attrib['end'])
        if chrono:
            t1_start_pos = int(t1_start_pos) + 1
            t1_end_pos = int(t1_end_pos) + 1
        duplicates = 0

        if t1_label in ["TIME", "FREQUENCY"]:
            continue
        else:
            for t2 in timex2:
                t2_start_pos, t2_end_pos = int(t2.attrib['start']), int(t2.attrib['end'])

                if len(set(range(t1_start_pos,t1_end_pos)).intersection(range(t2_start_pos,t2_end_pos))) > 0:
                    # we found an overlap
                    xml_str = tostring(t1).decode()
                    #print("Found overlap: " + xml_str)
                    duplicates = duplicates + 1
                    intersect.append(t1)
            if duplicates > 1:
                print("Multiple matches found to gold TIMEX! Num Duplicates: " + str(duplicates))
                print("Gold TIMEX: " + str(t1.attrib['text']) + "\nStart: " + str(t1_start_pos))
    return intersect


def writeXML(dir, text, timex):
    f = open(dir, mode='w')
    f.write("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n<ClinicalNarrativeTemporalAnnotation>\n<TEXT><![CDATA[")
    f.write(text)
    f.write("]]></TEXT>\n<TAGS>\n")
    for t1 in timex:

        #<TIMEX3 id="T8" start="602" end="610" text="two days" type="DURATION" val="P2D" mod="NA" />
        exp = '<TIMEX3 id=\"' + t1.attrib['id'] + \
                 '\" start=\"' + t1.attrib['start'] + \
                 '\" end=\"' + t1.attrib['end'] + \
                 '\" text=\"' + t1.attrib['text'] + \
                 '\" type=\"' + t1.attrib['type'] + \
                 '\" val=\"' + t1.attrib['val'] + \
                 '\" mod=\"' + t1.attrib['mod'] + \
                 '\" />\n'
        f.write(exp)

    f.write("</TAGS>\n</ClinicalNarrativeTemporalAnnotation>")

if __name__ == '__main__':

    ## Parse input arguments
    parser = argparse.ArgumentParser(
        description='Filter an i2b2 XML results file to only contain those timex expressions found in the reference/gold standard.')
    parser.add_argument('-r', metavar='referenceDir', type=str,
                        help='Path and name of directory where the reference XML files reside.',
                        required=True)
    parser.add_argument('-i', metavar='inputDir', type=str,
                        help='Path and directory name where the XML files needing to be filtered reside.',
                        required=True)
    parser.add_argument('-o', metavar='output', type=str,
                        help='Unique path to and name of output directory.',
                        required=False, default='./date-dur')
    parser.add_argument('-c', metavar='chrono', type=bool,
                        help='If processing Chrono output it is always off by one.',
                        required=False, default=False)

    args = parser.parse_args()


    ref_dir = args.r
    in_dir = args.i
    out_dir = args.o
    xml_filenames = [x for x in os.listdir(in_dir) if x.endswith('xml')]

    for xml in xml_filenames:
        print("PROCESSING>>> " + xml)
        patient_num = int(xml[:-4])
        #print("Patient Num: " + str(patient_num))
        in_xml_filepath = os.path.join(in_dir, xml)
        ref_xml_filepath = os.path.join(ref_dir, xml)

        in_text, in_timex = get_timex_list(in_xml_filepath)
        ref_text, ref_timex = get_timex_list(ref_xml_filepath)

        overlap_timex = intersect_timex(in_timex, ref_timex, chrono=args.c)

        #print("Number of overlaps: " + str(len(overlap_timex)))
        writeXML(os.path.join(out_dir, xml), in_text, overlap_timex)

    print("Filtering Completed!")