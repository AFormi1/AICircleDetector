using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AICircleDetector.AI
{
    public class BoundingBox
    {
        public int XMin { get; set; }   // Left coordinate
        public int YMin { get; set; }   // Top coordinate
        public int XMax { get; set; }   // Right coordinate
        public int YMax { get; set; }   // Bottom coordinate

        public string Label { get; set; }    // e.g. "circle"
        public float Confidence { get; set; } // Optional: detection confidence score

        public BoundingBox(int xmin, int ymin, int xmax, int ymax, string label = "circle", float confidence = 1.0f)
        {
            XMin = xmin;
            YMin = ymin;
            XMax = xmax;
            YMax = ymax;
            Label = label;
            Confidence = confidence;
        }

        // Optional: get width and height
        public int Width => XMax - XMin;
        public int Height => YMax - YMin;

        // Optional: get center point
        public (int X, int Y) Center => (XMin + Width / 2, YMin + Height / 2);
    }

}
