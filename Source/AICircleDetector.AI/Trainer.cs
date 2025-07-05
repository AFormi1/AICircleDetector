using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.NumPy;

namespace AICircleDetector.AI
{
    public static class Trainer
    {
       public static NDArray LoadImageAsNDArray(string path)
        {
            using var bmp = new Bitmap(path);
            var data = new float[bmp.Width, bmp.Height, 3];

            for (int y = 0; y < bmp.Height; y++)
            {
                for (int x = 0; x < bmp.Width; x++)
                {
                    var pixel = bmp.GetPixel(x, y);
                    data[x, y, 0] = pixel.R / 255f;
                    data[x, y, 1] = pixel.G / 255f;
                    data[x, y, 2] = pixel.B / 255f;
                }
            }

            return np.array(data).reshape(AIConfig.ImageShape);
        }
    }
}
