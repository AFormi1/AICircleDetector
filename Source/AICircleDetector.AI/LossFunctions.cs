using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras.Losses;
using static Tensorflow.Binding;

namespace AICircleDetector.AI
{
    public class BoundingBoxLoss : ILossFunc
    {
        public string Reduction => "sum";
        public string Name => "bounding_box_loss";

        public Tensor Call(Tensor y_true, Tensor y_pred, Tensor sample_weight = null)
        {
            return tf.reduce_mean(tf.square(y_true - y_pred));
        }
    }

    public class CircleCountLoss : ILossFunc
    {
        public string Reduction => "sum";
        public string Name => "circle_count_loss";

        public Tensor Call(Tensor y_true, Tensor y_pred, Tensor sample_weight = null)
        {
            // Optional: scale the true count to [0,1] if you use sigmoid in output
            return tf.reduce_mean(tf.square(y_true - y_pred));
        }
    }

    public class CombinedLoss : ILossFunc
    {
        public string Reduction => "sum";
        public string Name => "combined_loss";

        private readonly BoundingBoxLoss bboxLoss = new BoundingBoxLoss();
        private readonly CircleCountLoss countLoss = new CircleCountLoss();

        public Tensor Call(Tensor y_true, Tensor y_pred, Tensor sample_weight = null)
        {
            // Split y_true and y_pred
            // Assuming y_true = [bbox_labels, count_label]
            // and y_pred = [bbox_output, count_output]
            var y_true_list = tf.unstack(y_true, num: 2);  // or manually extract from NDArray[]
            var y_pred_list = tf.unstack(y_pred, num: 2);

            var bbox_loss = bboxLoss.Call(y_true_list[0], y_pred_list[0]);
            var count_loss = countLoss.Call(y_true_list[1], y_pred_list[1]);

            return bbox_loss + count_loss;
        }
    }
}
