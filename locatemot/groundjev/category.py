"""Native typed category decisions on the same OneJev engine as identity.

Large dynamic vocabularies use explicit hierarchical routing. The path
probability is not a calibrated detection score or a full flat distribution.
"""
import math


class NativeCategoryHead:
    def __init__(self, engine, group_size=128, batch_candidates=8):
        self.engine = engine
        self.group_size = group_size
        self.batch_candidates = batch_candidates

    def crop_sheet(self, image, candidates):
        from PIL import Image, ImageDraw, ImageOps
        sheet = Image.new('RGB', (4*192, max(1, math.ceil(len(candidates)/4))*216), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, candidate in enumerate(candidates):
            x, y = i%4*192, i//4*216
            draw.text((x+2, y+2), f'Object {i}', fill='black')
            crop = image.crop(tuple(candidate['box'])).convert('RGB')
            # KITTI/TAO small objects need visible crop evidence; enlarge using
            # real pixels, without synthesizing content or changing boxes.
            crop = ImageOps.contain(crop, (188, 188), method=Image.Resampling.BICUBIC)
            sheet.paste(crop, (x+2, y+22))
        return sheet

    def classify(self, image, candidates, vocabulary):
        from qev.schema import ChoiceQuestion, SystemOneRequest
        groups = [vocabulary[i:i+self.group_size] for i in range(0, len(vocabulary), self.group_size)]
        roots = {str(i): ', '.join(labels) for i, labels in enumerate(groups)}
        roots['UNKNOWN'] = 'No supplied category describes this object, or visual evidence is insufficient'
        outputs, reports = [], []
        for offset in range(0, len(candidates), self.batch_candidates):
            batch = candidates[offset:offset+self.batch_candidates]
            media = [{'type': 'image', 'image': self.crop_sheet(image, batch)}]
            state = {'object_crop_sheet': '<image:1>', 'object_indices': list(range(len(batch))),
                     'task': 'visual object category recognition; identity is a separate decision',
                     'class_groups': 'Disjoint input vocabulary groups; group membership is alphabetical/input order, not a semantic class',
                     'evidence': 'Use the image crops; detector query labels are not category evidence'}
            questions = {f'group_{i}': ChoiceQuestion(type='choice',
                instructions=f'Which category group contains the best visual category for object {i}? Check the listed class names, not the group number. Choose UNKNOWN if none fits.',
                criteria=roots) for i in range(len(batch))}
            response, root_meta = self.engine.decide(SystemOneRequest(state=state, model='jev-latest', questions=questions), media=media)
            leaf_questions = {}
            for i in range(len(batch)):
                root = response.answers[f'group_{i}']
                if root.choice != 'UNKNOWN':
                    labels = groups[int(root.choice)]
                    options = {str(j): label for j, label in enumerate(labels)}
                    options['UNKNOWN'] = 'None of these visual categories fits, or the crop is insufficient'
                    leaf_questions[f'class_{i}'] = ChoiceQuestion(type='choice',
                        instructions=f'What is the visual object category of object {i}? Select the actual category from this group. Do not use identity or detector query labels.', criteria=options)
            leaf_response, leaf_meta = (None, None)
            if leaf_questions:
                leaf_response, leaf_meta = self.engine.decide(SystemOneRequest(state=state, model='jev-latest', questions=leaf_questions), media=media)
            for i in range(len(batch)):
                root = response.answers[f'group_{i}']
                label, leaf, probability = '__unknown__', None, root.probabilities[root.choice]
                if root.choice != 'UNKNOWN':
                    leaf = leaf_response.answers[f'class_{i}']
                    probability *= leaf.probabilities[leaf.choice]
                    if leaf.choice != 'UNKNOWN':
                        label = groups[int(root.choice)][int(leaf.choice)]
                outputs.append({'label': label, 'probability': probability,
                    'root': root.model_dump(), 'leaf': leaf.model_dump() if leaf else None,
                    'native_probabilities': True, 'vocabulary_size': len(vocabulary),
                    'probability_kind': 'uncalibrated native hierarchical path; not detector confidence or flat global MAP'})
            reports.append({'offset': offset, 'objects': len(batch), 'root': root_meta, 'leaf': leaf_meta})
        return outputs, {'policy': 'native_hierarchical', 'all_vocabulary_classes_in_root': True,
            'vocabulary_size': len(vocabulary), 'group_size': self.group_size,
            'routing': 'highest-probability root then conditional leaf; explicit UNKNOWN retains box',
            'native_detection_score_unchanged': True, 'batches': reports}


class CategorizedDecision:
    """Compose category and identity decisions without changing association."""
    def __init__(self, decision, category_head):
        self.decision, self.category_head = decision, category_head

    def decide(self, image, candidates, tracks, timestamp, specification):
        decisions, lifecycle, metadata = self.decision.decide(image, candidates, tracks, timestamp, specification)
        categories, report = self.category_head.classify(image, candidates, specification['vocabulary'])
        for decision, category in zip(decisions, categories):
            decision['category_decision'] = category
        metadata['category_decision'] = report
        return decisions, lifecycle, metadata
