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

        public static string TrainingFolderName { get; private set; } = "TrainingData";
        public static string ImageFolderName { get; private set; } = "images";
        public static string AnnotationsFolderName { get; private set; } = "annotations";
        public static string LabelMapName { get; private set; } = "label_map.pbtxt";
        public static string TrainListName { get; private set; } = "train.txt";
        public static string ValListName { get; private set; } = "val.txt";
        public static string TrainValListName { get; private set; } = "trainval.txt";


        public static string TensorFlowModel { get; private set; } = "circle_count_model";
        public static int MaxCircles { get; private set; } = 10;
        public static int MinCircles { get; private set; } = 0;
        public static int ImageSize { get; private set; } = 128;
    }
}
