using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AICircleDetector.AI
{
    public static class AIConfig
    {

        //Docs:
        //https://github.com/SciSharp/TensorFlow.NET
        //https://github.com/SciSharp/TensorFlow.NET/wiki/Using-GPU-with-Tensorflow.NET


        public static string TensorFlowModel { get; private set; } = "circle_count_model";
        public static int MaxCircles { get; private set; } = 50;
        public static int ImageSize { get; private set; } = 128;
    }
}
