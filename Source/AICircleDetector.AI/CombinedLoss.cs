using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras.Losses;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace AICircleDetector.AI
{
    public class CombinedLoss : ILossFunc
    {
        private readonly ILossFunc _mse;

        public CombinedLoss()
        {
            // Use MeanSquaredError loss internally
            _mse = keras.losses.MeanSquaredError();
        }

        // You can set Reduction to null or "auto" depending on your preference or framework expectations
        public string Reduction => "auto";

        public string Name => "combined_loss";

        public Tensor Call(Tensor y_true, Tensor y_pred, Tensor sample_weight = null)
        {
            var maxBoxes = AIConfig.MaxCircles;
            var bbox_len = maxBoxes * 4;

            // Slice bbox and count parts from concatenated tensors
            var y_true_bbox = tf.slice(y_true, new[] { 0, 0 }, new[] { -1, bbox_len });
            var y_true_count = tf.slice(y_true, new[] { 0, bbox_len }, new[] { -1, 1 });

            var y_pred_bbox = tf.slice(y_pred, new[] { 0, 0 }, new[] { -1, bbox_len });
            var y_pred_count = tf.slice(y_pred, new[] { 0, bbox_len }, new[] { -1, 1 });

            // Reshape bbox tensors to (batch, maxBoxes, 4)
            y_true_bbox = tf.reshape(y_true_bbox, new[] { -1, maxBoxes, 4 });
            y_pred_bbox = tf.reshape(y_pred_bbox, new[] { -1, maxBoxes, 4 });

            // Compute MSE for bbox and count separately
            var bbox_loss = _mse.Call(y_true_bbox, y_pred_bbox, sample_weight);
            var count_loss = _mse.Call(y_true_count, y_pred_count, sample_weight);

            // Weight count loss higher (e.g. 5x)
            var total_loss = bbox_loss + 5 * count_loss;

            return total_loss;
        }
    }

}
