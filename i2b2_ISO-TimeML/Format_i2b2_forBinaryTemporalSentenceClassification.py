# format_i2b2_forBinaryTemporalSentenceClassification.py
# Amy Olex
# 5/8/21
## Modified from Emily Alsentzer's code to look for temporal annotations instead of events.
## Some of this is also from: https://colab.research.google.com/drive/1V1UCHDuGasUQe-D6DMOB1WsBAe_hmjEa#scrollTo=k3l98Y-lXp8H
#

import numpy as np
import os
from xml.etree.ElementTree import parse, tostring


def read_xml_file(xml_path, match_text=True, chrono=True):
    """
    Function to read in an i2b2 XML file and extract out the TIMEX annotations per character.

    :param xml_path: Path to all the XML files.
    :param match_text: Set to TRUE if the extracted text from the text should match the observed text per the coordinates.
    :return: Returns two character arrays. First the text by character,
             second the TIMEX labels for each character.
    """
    start_cdata = "<TEXT><![CDATA["
    end_cdata = "]]></TEXT>"

    # Pulling out text blob
    with open(xml_path, mode='r') as f:
        lines = f.readlines()
        # print(lines)
        text, in_text = [], False
        for line, l in enumerate(lines):
            if start_cdata in l:
                text.append(list(l[l.find(start_cdata) + len(start_cdata):]))
                in_text = True
            elif end_cdata in l:
                text.append(list(l[:l.find(end_cdata)]))
                break
            elif in_text:
                #                 if xml_path.endswith('180-03.xml') and '0808' in l and 'Effingham' in l:
                #                     print("Adjusting known error")
                #                     l = l[:9] + ' ' * 4 + l[9:]
                # #                 elif xml_path.endswith('188-05.xml') and 'Johnson & Johnson' in l:
                # #                     print("Adjusting known error")
                # #                     l = l.replace('&', 'and')
                text.append(list(l))

    pos_transformer = {}

    linear_pos = 1
    for line, sentence in enumerate(text):
        for char_pos, char in enumerate(sentence):
            pos_transformer[linear_pos] = (line, char_pos)
            linear_pos += 1

    # Parsing XML Tags
    try:
        xml_parsed = parse(xml_path)
    except:
        print(xml_path)
        raise

    tag_containers = xml_parsed.findall('TAGS')
    assert len(tag_containers) == 1, "Found multiple tag sets!"
    tag_container = tag_containers[0]

    timex_tags = tag_container.findall('TIMEX3')
    timex_labels = [['O'] * len(sentence) for sentence in text]
    for timex_tag in timex_tags:
        base_label = timex_tag.attrib['type']
        start_pos, end_pos, timex_text = timex_tag.attrib['start'], timex_tag.attrib['end'], timex_tag.attrib['text']
        if chrono:
            start_pos = int(start_pos) + 1
            end_pos = int(end_pos) + 1
        start_pos, end_pos = int(start_pos) + 1, int(end_pos)
        timex_text = ' '.join(timex_text.split())
        #         if event_text == "0808 O’neil’s Court":
        #             print("Adjusting known error")
        #             end_pos -= 4
        #         if event_text == 'Johnson and Johnson' and xml_path.endswith('188-05.xml'):
        #             print("Adjusting known error")
        #             event_text = 'Johnson & Johnson'

        (start_line, start_char), (end_line, end_char) = pos_transformer[start_pos], pos_transformer[end_pos]

        obs_text = []
        for line in range(start_line, end_line + 1):
            t = text[line]
            s = start_char if line == start_line else 0
            e = end_char if line == end_line else len(t)
            obs_text.append(''.join(t[s:e + 1]).strip())
        obs_text = ' '.join(obs_text)
        obs_text = ' '.join(obs_text.split())

        if '&apos;' in obs_text and '&apos;' not in timex_text:
            timex_text = timex_text.replace("'", "&apos;")

        if '&quot;' in obs_text and '&quot;' not in timex_text:
            timex_text = timex_text.replace('"', '&quot;')

        if match_text:
            if obs_text != timex_text:
                print("WARNING Text Mismatch: " + timex_text + " v " + obs_text + "\n" + str(xml_path))
            #assert obs_text == timex_text, (
            #        ("Texts don't match! %s v %s ..." % (timex_text, obs_text)) + '\n\n\n' + str(start_pos) + ' ' + str(end_pos) +
            #        ' ' + ' ' + str(s) + ' ' + str(e) + ' ' + str(xml_path))

        if base_label.strip() == '':
            continue

        timex_labels[end_line][end_char] = 'I-%s' % base_label
        timex_labels[start_line][start_char] = 'B-%s' % base_label

        for line in range(start_line, end_line + 1):
            t = text[line]
            s = start_char + 1 if line == start_line else 0
            e = end_char - 1 if line == end_line else len(t) - 1
            for k in range(s, e + 1):
                timex_labels[line][k] = 'I-%s' % base_label

    return text, timex_labels


def merge_into_words(text_by_char, all_labels_by_char):
    """

    :param text_by_char:
    :param all_labels_by_char:
    :return:
    """
    assert len(text_by_char) == len(all_labels_by_char), "Incorrect # of sentences!"

    N = len(text_by_char)

    text_by_word, all_labels_by_word = [], []

    for sentence_num in range(N):
        sentence_by_char = text_by_char[sentence_num]
        labels_by_char = all_labels_by_char[sentence_num]

        assert len(sentence_by_char) == len(labels_by_char), "Incorrect # of chars in sentence!"
        S = len(sentence_by_char)

        if labels_by_char == (['O'] * len(sentence_by_char)):
            sentence_by_word = ''.join(sentence_by_char).split()
            labels_by_word = ['O'] * len(sentence_by_word)
        else:
            sentence_by_word, labels_by_word = [], []
            text_chunks, labels_chunks = [], []
            s = 0
            for i in range(S):
                if i == S - 1:
                    text_chunks.append(sentence_by_char[s:])
                    labels_chunks.append(labels_by_char[s:])
                elif labels_by_char[i] == 'O':
                    continue
                else:
                    if i > 0 and labels_by_char[i - 1] == 'O':
                        text_chunks.append(sentence_by_char[s:i])
                        labels_chunks.append(labels_by_char[s:i])
                        s = i
                    if labels_by_char[i + 1] == 'O' or labels_by_char[i + 1][2:] != labels_by_char[i][2:]:
                        text_chunks.append(sentence_by_char[s:i + 1])
                        labels_chunks.append(labels_by_char[s:i + 1])
                        s = i + 1

            for text_chunk, labels_chunk in zip(text_chunks, labels_chunks):
                assert len(text_chunk) == len(labels_chunk), "Bad Chunking (len)"
                assert len(text_chunk) > 0, "Bad chunking (len 0)" + str(text_chunks) + str(labels_chunks)

                labels_set = set(labels_chunk)
                assert labels_set == {'O'} or (len(labels_set) <= 3 and 'O' not in labels_set), (
                        ("Bad chunking (contents) %s" % ', '.join(labels_set)) + '\n\n' + str(text_chunks) + '\n' + str(labels_chunks)
                )

                text_chunk_by_word = ''.join(text_chunk).split()
                W = len(text_chunk_by_word)
                if W == 0:
                    #                     assert labels_set == set(['O']), "0-word chunking and non-0 label!" + str(
                    #                         text_chunks) + str(labels_chunks
                    #                     )
                    continue

                if labels_chunk[0] == 'O':
                    labels_chunk_by_word = ['O'] * W
                elif W == 1:
                    labels_chunk_by_word = [labels_chunk[0]]
                elif W == 2:
                    labels_chunk_by_word = [labels_chunk[0], labels_chunk[-1]]
                else:
                    labels_chunk_by_word = [
                                               labels_chunk[0]
                                           ] + [labels_chunk[1]] * (W - 2) + [
                                               labels_chunk[-1]
                                           ]

                sentence_by_word.extend(text_chunk_by_word)
                labels_by_word.extend(labels_chunk_by_word)

        assert len(sentence_by_word) == len(labels_by_word), "Incorrect # of words in sentence!"

        if len(sentence_by_word) == 0: continue

        text_by_word.append(sentence_by_word)
        all_labels_by_word.append(labels_by_word)
    return text_by_word, all_labels_by_word


def reprocess_timex_labels(folders, base_path='.', event_tag_type='event', match_text=True, dev_set_size=None):
    """

    :param folders:
    :param base_path:
    :param event_tag_type:
    :param match_text:
    :param dev_set_size:
    :return:
    """
    all_texts_by_patient, all_labels_by_patient = {}, {}

    print("\nSTARTING REPROCESS TIMEX LABELS\n")

    for folder in folders:
        #print(folder)
        folder_dir = os.path.join(base_path, folder)
        xml_filenames = [x for x in os.listdir(folder_dir) if x.endswith('xml')]
        # print(xml_filenames)
        for xml_filename in xml_filenames:
            print("PROCESSING>>> " + xml_filename)
            patient_num = int(xml_filename[:-4])
            #print("Patient Num: " + str(patient_num))
            xml_filepath = os.path.join(folder_dir, xml_filename)
            # print(match_text)
            text_by_char, labels_by_char = read_xml_file(
                xml_filepath,
                match_text=match_text
            )
            text_by_word, labels_by_word = merge_into_words(text_by_char, labels_by_char)

            if patient_num not in all_texts_by_patient:
                all_texts_by_patient[patient_num] = []
                all_labels_by_patient[patient_num] = []

            all_texts_by_patient[patient_num].extend(text_by_word)
            all_labels_by_patient[patient_num].extend(labels_by_word)

    patients = set(all_texts_by_patient.keys())

    #print("My Patients: " + str(patients))

    if dev_set_size is None:
        train_patients, dev_patients = list(patients), []
    else:
        N_train = int(len(patients) * (1 - dev_set_size))
        patients_random = np.random.permutation(list(patients))
        train_patients = list(patients_random[:N_train])
        dev_patients = list(patients_random[N_train:])

    train_texts, train_labels, train_patient = [], [], []
    dev_texts, dev_labels, dev_patient = [], [], []

    for patient_num in train_patients:
        #print("Processing Patient #: " + str(patient_num))
        train_texts.extend(all_texts_by_patient[patient_num])
        train_labels.extend(all_labels_by_patient[patient_num])
        #print("All labels by patient: " + str(all_labels_by_patient[patient_num]))

        train_patient.extend([patient_num] * len(all_labels_by_patient[patient_num]))
        #print("Train_patient by patient: " + str( [patient_num] * len(all_labels_by_patient[patient_num]) ))

    for patient_num in dev_patients:
        dev_texts.extend(all_texts_by_patient[patient_num])
        dev_labels.extend(all_labels_by_patient[patient_num])
        dev_patient.extend([patient_num] * len(all_labels_by_patient[patient_num]))

    train_out_text_by_sentence = []
    #print("Writing out to file...")
    for text, labels, patient in zip(train_texts, train_labels, train_patient):
        #print("Writing patient #: " + str(patient))

        patient2list = [patient] * len(labels)
        #print("with text: " + str(text) + "\nand labels: " + str(labels) + "\nand patient2list: " + str(patient2list))

        train_out_text_by_sentence.append('\n'.join('%s %s %s' % x for x in zip(text, labels, patient2list)))
    dev_out_text_by_sentence = []
    for text, labels, patient in zip(dev_texts, dev_labels, dev_patient):
        patient2list = [patient] * len(labels)
        dev_out_text_by_sentence.append('\n'.join('%s %s %s' % x for x in zip(text, labels, patient2list)))

    return '\n\n'.join(train_out_text_by_sentence), '\n\n'.join(dev_out_text_by_sentence)






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    final_train_text, final_dev_text = reprocess_timex_labels(
        ['2012-07-15.original-annotation.release.SUBSET'],
        base_path='/Volumes/GoogleDrive/My Drive/PhD - NLP/Chrono_i2b2/i2b2TemporalRelations2012_AmyCopy/',
        dev_set_size=0.1, match_text=True
    )

    test_text, _ = reprocess_timex_labels(
        ['2012-08-08.test-data.event-timex-groundtruth.SUBSET/xml'],
        base_path='/Volumes/GoogleDrive/My Drive/PhD - NLP/Chrono_i2b2/i2b2TemporalRelations2012_AmyCopy/',
        match_text=False, dev_set_size=None
    )

    ## the following code just outputs the total number of each label instance
    labels = {}
    for s in final_train_text, final_dev_text, test_text:
        for line in s.split('\n'):
            if line == '': continue
            label = line.split()[-1]
            assert label == 'O' or label.startswith('B-') or label.startswith('I-'), "label wrong! %s" % label
            if label not in labels:
                labels[label] = 1
            else:
                labels[label] += 1

    with open('data/train_timex.tsv', mode='w') as f:
        f.write(final_train_text)
    with open('data/dev_timex.tsv', mode='w') as f:
        f.write(final_dev_text)
    with open('data/test_timex.tsv', mode='w') as f:
        f.write(test_text)


    test_cls = utils.convert_to_sentence_classification(test_text)
    final_dev_cls = utils.convert_to_sentence_classification(final_dev_text)
    final_train_cls = utils.convert_to_sentence_classification(final_train_text)

    with open('data/train_timex_binary.tsv',
              mode='w') as f:
        for s, l in final_train_cls:
            f.write(s + "\t" + str(l) + "\n")
    with open('data/dev_timex_binary.tsv',
              mode='w') as f:
        for s, l in final_dev_cls:
            f.write(s + "\t" + str(l) + "\n")
    with open('data/test_timex_binary.tsv',
              mode='w') as f:
        for s, l in test_cls:
            f.write(s + "\t" + str(l) + "\n")

    test_datedur = utils.convert_to_date_duration_locations(test_text)
    final_dev_datedur = utils.convert_to_date_duration_locations(final_dev_text)
    final_train_datedur = utils.convert_to_date_duration_locations(final_train_text)

    with open('data/train_timex_datedur.tsv',
              mode='w') as f:
        for s, dt, dr in final_train_datedur:
            f.write(s + "\t" + str(dt) + "\t" + str(dr) + "\n")
    with open('data/dev_timex_datedur.tsv',
              mode='w') as f:
        for s, dt, dr in final_dev_datedur:
            f.write(s + "\t" + str(dt) + "\t" + str(dr) + "\n")
    with open('data/test_timex_datedur.tsv',
              mode='w') as f:
        for s, dt, dr in test_datedur:
            f.write(s + "\t" + str(dt) + "\t" + str(dr) + "\n")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
