using CavakazeRenderer.CrossPlatformAPIManager;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace CavakazeRenderer.ThreadStarts
{
    class RenderingStart
    {
        private Timer updateImageTicker;
        public RenderingStart()
        {
            updateImageTicker = new Timer(UpdateImageThread);
        }

        private void UpdateImageThread(object state)
        {

        }

        public void RenderThreadStart()
        {
            updateImageTicker.Change(Timeout.Infinite, 1);
            //CudaAPI.StartRendering();
            updateImageTicker.Change(0, 0);

        }
    }
}
