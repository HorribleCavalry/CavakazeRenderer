using System;
using System.Runtime.InteropServices;


namespace CavakazeRenderer.CudaAPI
{
    class RenderManager
    {
        /// <summary>
        /// Allocate and open Debug console.
        /// </summary>
        [DllImport("CudaAPI", EntryPoint = "OpenDebugConsole", CallingConvention = CallingConvention.Cdecl)]
        extern public static void OpenDebugConsole();

        /// <summary>
        /// Free and close Debug Console.
        /// </summary>
        [DllImport("CudaAPI", EntryPoint = "CloseDebugConsole", CallingConvention = CallingConvention.Cdecl)]
        extern public static void CloseDebugConsole();

        /// <summary>
        /// Initialize the necessary resources.
        /// </summary>
        [DllImport("CudaAPI", EntryPoint = "InitializeResources", CallingConvention = CallingConvention.Cdecl)]
        extern public static void InitializeResources();

        [DllImport("CudaAPI", EntryPoint = "StartRendering", CallingConvention = CallingConvention.Cdecl)]
        extern public static void StartRendering();
    }
}
