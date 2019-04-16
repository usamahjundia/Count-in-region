import tensorflow as tf
import numpy as np
from object_detection.utils import visualization_utils, label_map_util, ops
import os

class ObjectDetector(object):
    def __init__(self,model_name):
        self.model_name = model_name
        self.graph = tf.Graph()
        self.num_class = 1
        self.initialize_graph()
        self.initialize_labels()
        self.session = None
    
    def __del__(self):
        if self.session is not None:
            self.session.close()

    def get_detection_dict(self,image,session):
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = session.run(tensor_dict,
                                feed_dict={image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict, image
    
    def get_centroids_bboxes_from_dict(self,detection_dict,height,width,threshold=0.5):
        boxes = detection_dict['detection_boxes']
        scores = detection_dict['detection_scores']
        boxes = boxes[np.where(scores > threshold)]
        centroids = [((b[2] + b[0])*height/2,(b[3]+b[1])*width/2) for b in boxes]
        return centroids, boxes
    
    def translate_boxes_to_xywh(self,boxes,height,width):
        results = [[] for _ in range(len(boxes))]
        for i, box in enumerate(boxes):
            newbox = [0 for i in range(4)]
            newbox[0] = box[1] * width
            newbox[1] = box[0] * height
            newbox[2] = (box[3] - box[1]) * width
            newbox[3] = (box[2] - box[0]) * height
            results[i] = newbox
        return results

    def detect_centroids(self,image,session,threshold=0.5):
        height,width = image.shape[:2]
        detection_dict,_ = self.get_detection_dict(image,session)
        centroids, boxes = self.get_centroids_bboxes_from_dict(detection_dict,height,width,threshold=threshold)
        boxes = self.translate_boxes_to_xywh(boxes,height,width) 
        return np.array(centroids,dtype=np.int32), np.array(boxes,dtype=np.int32)

    def initialize_graph(self):
        model_path = os.path.join(self.model_name,'frozen_inference_graph.pb')
        with self.graph.as_default():
            temp_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path,'rb') as f:
                ser_graph = f.read()
                temp_graph_def.ParseFromString(ser_graph)
                tf.import_graph_def(temp_graph_def,name='')
    
    def initialize_labels(self):
        path_to_label = os.path.join(self.model_name,'label.pbtxt')        
        label_map = label_map_util.load_labelmap(path=path_to_label)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.num_class, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
