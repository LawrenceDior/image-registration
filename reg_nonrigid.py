import hyperspy.api as hs
import numpy as np
import scipy
import SimpleITK as sitk
import skimage

from image_processing import apply_displacement_field, apply_displacement_field_sitk
from scipy import ndimage as ndi


def horn_schunck(data_in: np.array, alpha_squared: float, num_iterations: int, use_average=False):
    '''
    Horn-Schunck method for optical flow.
    Outputs a series of (vertical, horizontal) displacement field pairs obtained from a corresponding series of 2D images.
    
    The larger `alpha_squared` is, the more strongly departure from smoothness is penalised relative to rate of change of brightness.
    When `alpha_squared` = 1, both kinds of error are weighted equally.
    When `use_average` is True, each displacement field has the same (u,v) values for all pixels. The value used is the mean of the individual values.
    '''
    assert len(data_in.shape) == 3
    (num_images, height, width) = data_in.shape
    # Estimate derivatives of intensity wrt x, y and t
    I_x = ndi.filters.convolve(data_in, np.array([[[-1,-1],[1,1]],[[-1,-1],[1,1]]])*0.25)
    I_y = ndi.filters.convolve(data_in, np.array([[[-1,1],[-1,1]],[[-1,1],[-1,1]]])*0.25)
    I_t = ndi.filters.convolve(data_in, np.array([[[-1,-1],[-1,-1]],[[1,1],[1,1]]])*0.25)
    u = np.zeros(data_in.shape)
    v = np.zeros(data_in.shape)
    kernel = np.array([[1,2,1],[2,0,2],[1,2,1]])/12
    for t in range(num_images):
        u_t = np.zeros((height, width))
        v_t = np.zeros((height, width))
        for _ in range(num_iterations):
            u_mean = ndi.filters.convolve(u_t, kernel)
            v_mean = ndi.filters.convolve(v_t, kernel)
            I_coefficient = (I_x[t] * u_mean + I_y[t] * v_mean + I_t[t]) / (alpha_squared + I_x[t]**2 + I_y[t]**2)
            u_t = u_mean - I_x[t] * I_coefficient
            v_t = v_mean - I_y[t] * I_coefficient
            if (np.array_equal(u_t, u[t]) and np.array_equal(v_t, v[t])):
                break
            u[t] = u_t
            v[t] = v_t
        if (use_average):
            u_avg = u[t].mean()
            v_avg = v[t].mean()
            u[t] = np.ones(u[t].shape) * u_avg
            v[t] = np.ones(v[t].shape) * v_avg
    return (u,v)


def horn_schunck_signal(signal_in: hs.signals.Signal2D, alpha_squared: float, num_iterations: int, use_average=False):
    '''
    Horn-Schunck method for optical flow.
    Outputs a series of (vertical, horizontal) displacement field pairs obtained from a corresponding series of 2D images.
    
    The larger `alpha_squared` is, the more strongly departure from smoothness is penalised relative to rate of change of brightness.
    When `alpha_squared` = 1, both kinds of error are weighted equally.
    When `use_average` is True, each displacement field has the same (u,v) values for all pixels. The value used is the mean of the individual values.
    '''
    num_images = signal_in.data.shape[0]
    image_shape = (signal_in.data.shape[1], signal_in.data.shape[2])
    # Estimate derivatives of intensity wrt x, y and t
    I_x = ndi.filters.convolve(signal_in.data, np.array([[[-1,-1],[1,1]],[[-1,-1],[1,1]]])*0.25)
    I_y = ndi.filters.convolve(signal_in.data, np.array([[[-1,1],[-1,1]],[[-1,1],[-1,1]]])*0.25)
    I_t = ndi.filters.convolve(signal_in.data, np.array([[[-1,-1],[-1,-1]],[[1,1],[1,1]]])*0.25)               
    u = np.zeros(signal_in.data.shape)
    v = np.zeros(signal_in.data.shape)
    kernel = np.array([[1,2,1],[2,0,2],[1,2,1]])/12
    for t in range(num_images):
        u_t = np.zeros(signal_in.data[t].shape)
        v_t = np.zeros(signal_in.data[t].shape)
        for _ in range(num_iterations):
            u_mean = ndi.filters.convolve(u_t, kernel)
            v_mean = ndi.filters.convolve(v_t, kernel)
            I_coefficient = (I_x[t] * u_mean + I_y[t] * v_mean + I_t[t]) / (alpha_squared + I_x[t]**2 + I_y[t]**2)
            u_t = u_mean - I_x[t] * I_coefficient
            v_t = v_mean - I_y[t] * I_coefficient
            if (np.array_equal(u_t, u[t]) and np.array_equal(v_t, v[t])):
                break
            u[t] = u_t
            v[t] = v_t
        if (use_average):
            u_avg = u[t].mean()
            v_avg = v[t].mean()
            u[t] = np.ones(u[t].shape) * u_avg
            v[t] = np.ones(v[t].shape) * v_avg
    return (u,v)


def displace_signal_using_horn_schunck_repeatedly(signal_in: hs.signals.Signal2D, alpha_squared=1.0, num_iterations_outer=5, num_iterations_inner=50, use_average=False, use_sitk=True):
    '''
    Applies the Horn-Schunck optical flow method repeatedly to an image stack. The number of times Horn-Schunck is repeated is determined by the num_iterations_outer parameter. The num_iterations_inner parameter of this method corresponds to the num_iterations parameter of horn_schunck.
    '''
    signal_out = signal_in
    for _ in range(num_iterations_outer):
        [us, vs] = horn_schunck(signal_out, alpha_squared, num_iterations_inner, use_average)
        data_out = np.empty(signal_in.data.shape)
        for t in range(signal_in.data.shape[0]):
            data_out[t] = signal_out.data[t]
            if (t < signal_in.data.shape[0] - 1):
                for t2 in range(t, signal_in.data.shape[0]-1):
                    if use_sitk:
                        data_out[t] = apply_displacement_field_sitk(np.array([us[t2], vs[t2]]), data_out[t])
                    else:
                        [data_out[t], _] = apply_displacement_field(np.array([us[t2], vs[t2]]), data_out[t])
        signal_out = hs.signals.Signal2D(data_out)
    return signal_out


def nonrigid_pyramid(signal_in: hs.signals.Signal2D, num_levels=3, max_image_dimension=64, registration_method=horn_schunck, reg_args=None, reg_kwargs={}):
    '''
    TODO: Finish implementing this function.
    Applies a given non-rigid registration algorithm to `signal_in`.
    If `num_levels` is greater than 1, a pyramid strategy is applied: the registration algorithm is initially performed on a downsampled version of `signal_in`, such that a coarse, approximate displacement field is obtained. The displacement field is applied to `signal_in` and the process is repeated on the result, this time less severely downsampled. Eventually, the registration method is applied to an image stack at the original resolution. By this point the remaining corrections needed should be small, so the registration method should cope better and produce a more sensible final result.
    If `max_image_dimension` is not equal to -1, `signal_in` is downsampled to such a size that the maximum of its height and width is equal to `max_image_dimension`. The function then proceeds as though that is the original size of the images in `signal_in`.
    `reg_args` and `reg_kwargs` contain any further arguments to be passed to `registration_method`.
    '''
    assert len(signal_in.data.shape) == 3
    (num_images, height, width) = signal_in.data.shape
    
    # Ensure the maximum dimension of images to be processed is no greater than max_image_dimension.
    height_resized = height
    width_resized = width
    if max_image_dimension == -1:
        max_image_dimension = max(height, width)
    else:
        assert max_image_dimension > 0
    resize_image = height > max_image_dimension or width > max_image_dimension
    if resize_image:
        if height > width:
            height_resized = max_image_dimension
            width_resized = int(max_image_dimension * width / height)
        else:
            width_resized = max_image_dimension
            height_resized = int(max_image_dimension * height / width)
        print("height_resized = " + str(height_resized) + ", width_resized = " + str(width_resized))
        
    us = np.zeros(data_in.shape)
    vs = np.zeros(data_in.shape)
    
    images_resized = np.zeros((num_images, height_resized, width_resized))
    if resize_image:
        for t in range(num_images):
            images_resized[t] += skimage.transform.resize(signal_in.data[t], (height_resized, width_resized), mode='reflect', anti_aliasing=True)
    else:
        for t in range(num_images):
            images_resized[t] += signal_in.data[t]
    
    # Pyramid loop
    for n in range(num_levels):
        power_of_2 = num_levels - 1 - n
        print("power_of_2 = " + str(power_of_2))
        (new_height, new_width) = (height_resized//(2**power_of_2), width_resized//(2**power_of_2))
        if new_height < 2 or new_width < 2:
            continue
        new_us = np.zeros((num_images, new_height, new_width))
        new_vs = np.zeros((num_images, new_height, new_width))
        
        images_downsampled = np.zeros((num_images, new_height, new_width))
        # TODO: apply displacement field (us, vs) to signal_in.data
        if power_of_2 > 0:
            for t in range(num_images):
                images_downsampled[t] += skimage.transform.resize(signal_in.data[t], (new_height, new_width), mode='reflect', anti_aliasing=True)
        else:
            for t in range(num_images):
                images_downsampled[t] += images_resized[t]
        
        # Estimate displacement fields
        if reg_args is None:
            (new_us, new_vs) = registration_method(images_downsampled, **reg_kwargs)
        else:
            (new_us, new_vs) = registration_method(images_downsampled, *reg_args, **reg_kwargs)
        
        # TODO: interpolate (new_us, new_vs) to give (us, vs)

def nonrigid(im_stack, demons_it = 20, filter_size = 5.0, max_it = 3, default_pixel_value = 100):
    '''
    Nonrigid optical flow method based on SmartAlign.
    Author: qzo13262
    Modified by fqj69741
    '''
    
    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations( demons_it )
    # Standard deviation for Gaussian smoothing of displacement field
    demons.SetStandardDeviations( filter_size )
    
    for j in range(max_it):
        #Get stack average
        # This will be treated as a reference image. Each 'moving' image will be transformed to align with av_im.
        # The reference image is recalculated max_it times.
        av_im = sitk.GetImageFromArray(np.float32(sum(im_stack)/len(im_stack))) #Faster than numpy.mean for small arrays?
        
        #Get gradient images of stack average
        '''avgrad = np.gradient(av_im)
        normavgradx = avgrad[0]/np.sqrt(np.square(avgrad[0])+np.square(avgrad[1]))
        normavgrady = avgrad[1]/np.sqrt(np.square(avgrad[0])+np.square(avgrad[1]))'''
        
        out_stack = []
        
        for i in range(len(im_stack)):
            
            #print(im_stack[i])
            
            # Load up the next 'moving' image from the image stack.
            moving = sitk.GetImageFromArray(np.float32(im_stack[i]))
            
            # Use the 'demons' method to obtain a displacement field that transforms the moving image
            # to align with the reference image.
            displacementField = demons.Execute( av_im, moving )
            
            dispfield = sitk.GetArrayFromImage(displacementField)
            #print(dispfield)
            
            '''if dispconstraint == 'rwo-locked':
                disp_contrained = '''
            
            # Convert displacementField to a transformation that can be appplied to the moving image via a ResampleImageFilter.
            outTx = sitk.DisplacementFieldTransform( displacementField )
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(av_im);
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(default_pixel_value)
            resampler.SetTransform(outTx)
            
            # Append the transformed image to out_stack.
            out_stack.append(sitk.GetArrayFromImage(resampler.Execute(moving)))
            
            '''grad = np.gradient(im_stack[i])
            normgradx = grad[0]/np.sqrt(np.square(grad[0])+np.square(grad[1]))
            normgrady = grad[1]/np.sqrt(np.square(grad[0])+np.square(grad[1]))
            
            transform_x = (av_im-im_stack[i])*(normgradx+normavgradx)
            transform_y = (av_im-im_stack[i])*(normgrady+normavgrady)
            
            tx_filtered = ndimage.filters.gaussian_filter(transform_x,sigma=10)
            ty_filtered = ndimage.filters.gaussian_filter(transform_y,sigma=10)
            
            t_filtered = np.stack((ty_filtered,tx_filtered))
            
            im_t = transform.warp(im_stack[i],t_filtered)'''
            
        im_stack = out_stack
        
        #dispfield = sitk.GetArrayFromImage(displacementField)
        #print(dispfield)
        
        # dispfield is out of scope here, but its most recent value is retained.
        max_disp = np.max(dispfield)
            
        # Terminate early if no part of the (last) image has been displaced by more than 1 pixel.
        #if max_disp < 1.0:
        #    print("NRR stopped after "+str(j)+" iterations.")
        #    break
    
    return(out_stack)