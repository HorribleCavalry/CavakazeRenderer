using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Threading;
using CavakazeRenderer.IDGenerator;
using CavakazeRenderer.CrossPlatformAPIManager;

namespace CavakazeRenderer.Managers
{
    class RenderManager: Manager
    {
        ThreadStarts.RenderingStart renderThreadStart;
        private Dictionary<ulong, Thread> threadPool;

        private IDGenerator.IDGenerator IDGenerator;

        private ulong renderThreadID;

        public RenderManager()
        {
            Initialize();
        }

        public void Initialize()
        {
            threadPool = new Dictionary<ulong, Thread>();
            IDGenerator = new OrderedIDGenerator();
            renderThreadID = IDGenerator.AllocateID();
            renderThreadStart = new ThreadStarts.RenderingStart();
            threadPool.Add(renderThreadID, new Thread(renderThreadStart.RenderThreadStart));
        }

        unsafe public void StartRendering(int width, int height, IntPtr imagePtr)
        {
            CudaAPI.StartRendering(width, height, (void*)imagePtr);

            //Thread renderThread = threadPool[renderThreadID];
            //switch (renderThread.ThreadState)
            //{
            //    case ThreadState.Running:
            //        break;
            //    case ThreadState.StopRequested:
            //        break;
            //    case ThreadState.SuspendRequested:
            //        break;
            //    case ThreadState.Background:
            //        break;
            //    case ThreadState.Unstarted:
            //        renderThread.Start();
            //        break;
            //    case ThreadState.Stopped:
            //        renderThread.Start();
            //        break;
            //    case ThreadState.WaitSleepJoin:
            //        break;
            //    case ThreadState.Suspended:
            //        break;
            //    case ThreadState.AbortRequested:
            //        break;
            //    case ThreadState.Aborted:
            //        break;
            //    default:
            //        break;
            //}
        }

        public void Release()
        {

        }
    }
}
