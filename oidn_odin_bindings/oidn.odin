
package oidn

import "core:c"

CudaStream :: distinct rawptr
HipStream :: distinct rawptr

MTLDevice_id :: distinct rawptr
MTLCommandQueue_id :: distinct rawptr
MTLBuffer_id :: distinct rawptr

// -------------------------------------------------------------------------------------------------
// Physical Device
// -------------------------------------------------------------------------------------------------

UUID_SIZE :: 16 // size of a universally unique identifier (UUID) of a physical device
LUID_SIZE :: 8  // size of a locally unique identifier (LUID) of a physical device

foreign import oidn_clib "OpenImageDenoise.lib"

@(default_calling_convention="c", link_prefix="oidn")
foreign oidn_clib
{
    // Returns the number of supported physical devices.
    GetNumPhysicalDevices :: proc() -> c.int ---

    // Gets a boolean parameter of the physical device.
    GetPhysicalDeviceBool :: proc(physicalDeviceID: c.int, name: cstring) -> c.bool ---

    // Gets an integer parameter of the physical device.
    GetPhysicalDeviceInt :: proc(physicalDeviceID: c.int, name: cstring) -> c.int ---

    // Gets a string parameter of the physical device.
    GetPhysicalDeviceString :: proc(physicalDeviceID: c.int, name: cstring) -> cstring ---

    // Gets an opaque data parameter of the physical device.
    GetPhysicalDeviceData :: proc(physicalDeviceID: c.int, name: cstring, byteSize: ^c.size_t) -> rawptr ---
}

// Gets an unsigned integer parameter of the physical device.
GetPhysicalDeviceUInt :: proc(physicalDeviceID: c.int, name: cstring) -> c.uint
{
    return c.uint(GetPhysicalDeviceInt(physicalDeviceID, name))
}

// -------------------------------------------------------------------------------------------------
// Device
// -------------------------------------------------------------------------------------------------

// Device types
DeviceType :: enum c.int
{
    DEFAULT = 0, // select device automatically

    CPU   = 1, // CPU device
    SYCL  = 2, // SYCL device
    CUDA  = 3, // CUDA device
    HIP   = 4, // HIP device
    METAL = 5, // Metal device
}

// Error codes
Error :: enum c.int
{
    NONE                 = 0, // no error occurred
    UNKNOWN              = 1, // an unknown error occurred
    INVALID_ARGUMENT     = 2, // an invalid argument was specified
    INVALID_OPERATION    = 3, // the operation is not allowed
    OUT_OF_MEMORY        = 4, // not enough memory to execute the operation
    UNSUPPORTED_HARDWARE = 5, // the hardware (e.g. CPU) is not supported
    CANCELLED            = 6, // the operation was cancelled by the user
}

// Error callback function
ErrorFunction :: #type proc "c"(userPtr: rawptr, code: Error, message: cstring)

// Device handle
Device :: distinct rawptr

@(default_calling_convention="c", link_prefix="oidn")
foreign oidn_clib
{
    // Returns whether the CPU device is supported.
    IsCPUDeviceSupported :: proc() -> c.bool ---

    // Returns whether the specified CUDA device is supported.
    IsCUDADeviceSupported :: proc(deviceID: c.int) -> c.bool ---

    // Returns whether the specified HIP device is supported.
    IsHIPDeviceSupported :: proc(deviceID: c.int) -> c.bool ---

    // Returns whether the specified Metal device is supported.
    IsMetalDeviceSupported :: proc(device: MTLDevice_id) -> c.bool ---

    // Creates a device of the specified type.
    NewDevice :: proc(type: DeviceType) -> Device ---

    // Creates a device from a physical device specified by its ID (0 to GetNumPhysicalDevices()-1).
    NewDeviceByID :: proc(physicalDeviceID: c.int) -> Device ---

    // Creates a device from a physical device specified by its UUID.
    NewDeviceByUUID :: proc(uuid: rawptr) -> Device ---

    // Creates a device from a physical device specified by its LUID.
    NewDeviceByLUID :: proc(void: rawptr) -> Device ---

    // Creates a device from a physical device specified by its PCI address.
    NewDeviceByPCIAddress :: proc(pciDomain: c.int, pciBus: c.int, pciDevice: c.int, pciFunction: c.int) -> Device ---

    // Creates a device from the specified pairs of CUDA device IDs and streams (null stream
    // corresponds to the default stream). Currently only one device ID/stream is supported.
    NewCUDADevice :: proc(deviceIDs: [^]c.int, streams: [^]CudaStream, numPairs: c.int) -> Device ---

    // Creates a device from the specified pairs of HIP device IDs and streams (null stream
    // corresponds to the default stream). Currently only one device ID/stream is supported.
    NewHIPDevice :: proc(deviceIDs: [^]c.int, streams: [^]HipStream, numPairs: c.int) -> Device ---

    // Creates a device from the specified list of Metal command queues.
    // Currently only one queue is supported.
    NewMetalDevice :: proc(commandQueues: [^]MTLCommandQueue_id, numQueues: c.int) -> Device ---

    // Retains the device (increments the reference count).
    RetainDevice :: proc(device: Device) ---

    // Releases the device (decrements the reference count).
    ReleaseDevice :: proc(device: Device) ---

    // Sets a boolean parameter of the device.
    SetDeviceBool :: proc(device: Device, name: cstring, value: c.bool) ---

    // Sets an integer parameter of the device.
    SetDeviceInt :: proc(device: Device, name: cstring, value: c.int) ---

    // Gets a boolean parameter of the device.
    GetDeviceBool :: proc(device: Device, name: cstring) -> c.bool ---

    // Gets an integer parameter of the device.
    GetDeviceInt :: proc(device: Device, name: cstring) -> c.int ---

    // Sets the error callback function of the device.
    SetDeviceErrorFunction :: proc(device: Device, func: ErrorFunction, userPtr: rawptr) ---

    // Returns the first unqueried error code stored in the device for the current thread, optionally
    // also returning a string message (if not NULL), and clears the stored error. Can be called with
    // a NULL device as well to check for per-thread global errors (e.g. why a device creation or
    // physical device query has failed).
    GetDeviceError :: proc(device: Device, outMessage: ^cstring) -> Error ---

    // Commits all previous changes to the device.
    // Must be called before first using the device (e.g. creating filters).
    CommitDevice :: proc(device: Device) ---

    // Waits for all asynchronous operations running on the device to complete.
    SyncDevice :: proc(device: Device) ---
}

// Sets an unsigned integer parameter of the device.
SetDeviceUInt :: proc(device: Device, name: cstring, value: u32)
{
    SetDeviceInt(device, name, c.int(value))
}

// Gets an unsigned integer parameter of the device.
GetDeviceUInt :: proc(device: Device, name: cstring) -> u32
{
    return u32(GetDeviceInt(device, name))
}

// -------------------------------------------------------------------------------------------------
// Buffer
// -------------------------------------------------------------------------------------------------

// Formats for images and other data stored in buffers
Format :: enum c.int
{
    UNDEFINED = 0,

    // 32-bit single-precision floating-point scalar and vector formats
    FLOAT  = 1,
    FLOAT2,
    FLOAT3,
    FLOAT4,

    // 16-bit half-precision floating-point scalar and vector formats
    HALF  = 257,
    HALF2,
    HALF3,
    HALF4,
}

// Storage modes for buffers
Storage :: enum c.int
{
    UNDEFINED = 0,

    // stored on the host, accessible by both host and device
    HOST      = 1,

    // stored on the device, *not* accessible by the host
    DEVICE    = 2,

    // automatically migrated between host and device, accessible by both
    // *not* supported by all devices, "managedMemorySupported" device parameter should be checked
    MANAGED   = 3,
}

// External memory type flags
ExternalMemoryTypeFlags :: distinct bit_set[ExternalMemoryTypeFlag; c.int]
ExternalMemoryTypeFlag :: enum c.int
{
    // opaque POSIX file descriptor handle
    OPAQUE_FD = 0,

    // file descriptor handle for a Linux dma_buf
    DMA_BUF = 1,

    // NT handle
    OPAQUE_WIN32 = 2,

    // global share (KMT) handle
    OPAQUE_WIN32_KMT = 3,

    // NT handle returned by IDXGIResource1::CreateSharedHandle referring to a Direct3D 11 texture
    // resource
    D3D11_TEXTURE = 4,

    // global share (KMT) handle returned by IDXGIResource::GetSharedHandle referring to a Direct3D 11
    // texture resource
    D3D11_TEXTURE_KMT = 5,

    // NT handle returned by IDXGIResource1::CreateSharedHandle referring to a Direct3D 11 resource
    D3D11_RESOURCE = 6,

    // global share (KMT) handle returned by IDXGIResource::GetSharedHandle referring to a Direct3D 11
    // resource
    D3D11_RESOURCE_KMT = 7,

    // NT handle returned by ID3D12Device::CreateSharedHandle referring to a Direct3D 12 heap
    // resource
    D3D12_HEAP = 8,

    // NT handle returned by ID3D12Device::CreateSharedHandle referring to a Direct3D 12 committed
    // resource
    D3D12_RESOURCE = 9,
}

EXTERNAL_MEMORY_TYPE_FLAGS_NONE :: ExternalMemoryTypeFlags {}

// Buffer handle
Buffer :: distinct rawptr

@(default_calling_convention="c", link_prefix="oidn")
foreign oidn_clib
{
    // Creates a buffer accessible to both the host and device.
    NewBuffer :: proc(device: Device, byteSize: c.size_t) -> Buffer ---

    // Creates a buffer with the specified storage mode.
    NewBufferWithStorage :: proc(device: Device, byteSize: c.size_t, storage: Storage) -> Buffer ---

    // Creates a shared buffer from memory allocated and owned by the user and accessible to the device.
    NewSharedBuffer :: proc(device: Device, devPtr: rawptr, byteSize: c.size_t) -> Buffer ---

    // Creates a shared buffer by importing external memory from a POSIX file descriptor.
    NewSharedBufferFromFD :: proc(device: Device,
                                  fdType: ExternalMemoryTypeFlags,
                                  fd: c.int, byteSize: c.size_t) -> Buffer ---

    // Creates a shared buffer by importing external memory from a Win32 handle.
    NewSharedBufferFromWin32Handle :: proc(device: Device,
                                           handleType: ExternalMemoryTypeFlags,
                                           handle: rawptr, name: rawptr, byteSize: c.size_t) -> Buffer ---

    // Creates a shared buffer from a Metal buffer.
    // Only buffers with shared or private storage and hazard tracking are supported.
    NewSharedBufferFromMetal :: proc(device: Device, buffer: MTLBuffer_id) -> Buffer ---

    // Gets the size of the buffer in bytes.
    GetBufferSize :: proc(buffer: Buffer) -> c.size_t ---

    // Gets the storage mode of the buffer.
    GetBufferStorage :: proc(buffer: Buffer) -> Storage ---

    // Gets a pointer to the buffer data, which is accessible to the device but not necessarily to
    // the host as well, depending on the storage mode. Null pointer may be returned if the buffer
    // is empty or getting a pointer to data with device storage is not supported by the device.
    GetBufferData :: proc(buffer: Buffer) -> rawptr ---

    // Copies data from a region of the buffer to host memory.
    ReadBuffer :: proc(buffer: Buffer, byteOffset: c.size_t, byteSize: c.size_t, dstHostPtr: rawptr) ---

    // Copies data from a region of the buffer to host memory asynchronously.
    ReadBufferAsync :: proc(buffer: Buffer,
                            byteOffset: c.size_t, byteSize: c.size_t, dstHostPtr: rawptr) ---

    // Copies data to a region of the buffer from host memory.
    WriteBuffer :: proc(buffer: Buffer,
                        byteOffset: c.size_t, byteSize: c.size_t, srcHostPtr: rawptr) ---

    // Copies data to a region of the buffer from host memory asynchronously.
    WriteBufferAsync :: proc(buffer: Buffer,
                             byteOffset: c.size_t, byteSize: c.size_t, srcHostPtr: rawptr) ---

    // Retains the buffer (increments the reference count).
    RetainBuffer :: proc(buffer: Buffer) ---

    // Releases the buffer (decrements the reference count).
    ReleaseBuffer :: proc(buffer: Buffer) ---
}

// -------------------------------------------------------------------------------------------------
// Filter
// -------------------------------------------------------------------------------------------------

// Filter quality/performance modes
Quality :: enum c.int
{
    DEFAULT  = 0, // default quality

    FAST     = 4, // high performance (for interactive/real-time preview rendering)
    BALANCED = 5, // balanced quality/performance (for interactive/real-time rendering)
    HIGH     = 6, // high quality (for final-frame rendering)
}

// Progress monitor callback function
ProgressMonitorFunction :: #type proc(userPtr: rawptr, n: f64)

// Filter handle
Filter :: distinct rawptr

@(default_calling_convention="c", link_prefix="oidn")
foreign oidn_clib
{
    // Creates a filter of the specified type (e.g. "RT").
    NewFilter :: proc(device: Device, type: cstring) -> Filter ---

    // Retains the filter (increments the reference count).
    RetainFilter :: proc(filter: Filter) ---

    // Releases the filter (decrements the reference count).
    ReleaseFilter :: proc(filter: Filter) ---

    // Sets an image parameter of the filter with data stored in a buffer.
    // If pixelByteStride and/or rowByteStride are zero, these will be computed automatically.
    SetFilterImage :: proc(filter: Filter, name: cstring,
                           buffer: Buffer, format: Format,
                           width: c.size_t, height: c.size_t,
                           byteOffset: c.size_t = 0,
                           pixelByteStride: c.size_t = 0, rowByteStride: c.size_t = 0) ---

    // Sets an image parameter of the filter with data owned by the user and accessible to the device.
    // If pixelByteStride and/or rowByteStride are zero, these will be computed automatically.
    SetSharedFilterImage :: proc(filter: Filter, name: cstring,
                                 devPtr: rawptr, format: Format,
                                 width: c.size_t, height: c.size_t,
                                 byteOffset: c.size_t = 0,
                                 pixelByteStride: c.size_t = 0, rowByteStride: c.size_t = 0) ---

    // Unsets an image parameter of the filter that was previously set.
    UnsetFilterImage :: proc(filter: Filter, name: cstring) ---

    // Sets an opaque data parameter of the filter owned by the user and accessible to the host.
    SetSharedFilterData :: proc(filter: Filter, name: cstring,
                                hostPtr: rawptr, byteSize: c.size_t) ---

    // Notifies the filter that the contents of an opaque data parameter has been changed.
    UpdateFilterData :: proc(filter: Filter, name: cstring) ---

    // Unsets an opaque data parameter of the filter that was previously set.
    UnsetFilterData :: proc(filter: Filter, name: cstring) ---

    // Sets a boolean parameter of the filter.
    SetFilterBool :: proc(filter: Filter, name: cstring, value: c.bool) ---

    // Gets a boolean parameter of the filter.
    GetFilterBool :: proc(filter: Filter, name: cstring) -> c.bool ---

    // Sets an integer parameter of the filter.
    SetFilterInt :: proc(filter: Filter, name: cstring, value: c.int) ---

    // Gets an integer parameter of the filter.
    GetFilterInt :: proc(filter: Filter, name: cstring) -> c.int ---

    // Sets a float parameter of the filter.
    SetFilterFloat :: proc(filter: Filter, name: cstring, value: f32) ---

    // Gets a float parameter of the filter.
    GetFilterFloat :: proc(filter: Filter, name: cstring) -> f32 ---

    // Sets the progress monitor callback function of the filter.
    SetFilterProgressMonitorFunction :: proc(filter: Filter,
                                             func: ProgressMonitorFunction, userPtr: rawptr) ---

    // Commits all previous changes to the filter.
    // Must be called before first executing the filter.
    CommitFilter :: proc(filter: Filter) ---

    // Executes the filter.
    ExecuteFilter :: proc(filter: Filter) ---

    // Executes the filter asynchronously.
    ExecuteFilterAsync :: proc(filter: Filter) ---
}
