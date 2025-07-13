using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AICircleDetector.AI
{
    public class PredictionResult
    {
        public string Result { get; set; } = string.Empty;
        public List<Circle> Circles { get; set; } = new();
    }
}
