import theano
from theano import  tensor as T
from theano.tensor.nnet import conv2d
import numpy
rng =   numpy.random.RandomState(23455)


def testConv2d():
    input = T.tensor4(name='input')
    w_shp = (2,3,3,3)
    w_bound = numpy.sqrt( 3 * 3 * 3)
    W = theano.shared(numpy.asarray(
                      rng.uniform(
                          low  = -1.0 / w_bound,
                          high = 1.0 / w_bound,
                          size = w_shp),
                       dtype= theano.config.floatX),name='W')

	 # use  defined W not the 	 rng.uniform
    W = theano.shared(value= numpy.array([ [ [ [[.0],[.0],[.0]],[[.0],[1.0],[.0]],[[.0],[0.0],[.0]]],
                                            [[[.0], [.0], [.0]], [[.0], [1.0], [0.0]], [[.0], [0.0], [.0]]],
                                            [[[.0], [.0], [.0]], [[.0], [.0], [.0]], [[.0], [0.0], [.0]]] ],
                                           [[[[.0], [.0], [.0]], [[.0], [.0], [.0]], [[.0], [0.0], [.0]]],
                                            [[[.0], [.0], [.0]], [[.0], [0.0], [.0]], [[.0], [0.0], [.0]]],
                                            [[[.0], [.0], [.0]], [[.0], [.0], [.0]], [[.0], [0.0], [.0]]]] ]
                                         ).reshape(w_shp) ,name = 'W')
    b_shp =(2,)
    b = theano.shared(numpy.asarray(
                      rng.uniform(
                          low= -.5,
                          high = .5,
                          size= b_shp
                      ),
                      dtype = theano.config.floatX),name='b')
    con_out = conv2d(input,W)
    #output  = T.nnet.sigmoid(con_out + b.dimshuffle('x',0,'x','x'))
    output = con_out
    #output  = T.nnet.sigmoid(con_out)

    import pylab
    from PIL import  Image
    img = Image.open(open('3wolfmoon.jpg'))
    img = numpy.asarray(img,dtype='float64') / 256.
    img_ = img.transpose(2,0,1).reshape(1,3,639,516)
    con_out = conv2d(img_, W)
    f = theano.function(inputs = [input],outputs=output)


    import pylab
    from PIL import  Image
    img = Image.open(open('3wolfmoon.jpg'))
    img = numpy.asarray(img,dtype='float64') / 256.
    img_ = img.transpose(2,0,1).reshape(1,3,639,516)
    img_1 = img_.copy()
    img_1[0, 0, :, :] = 0.72
    img_1[0,1,:,:] = 0.72
    img_1[0, 2, :, :] = 0.0
    filtered_img = f(img_)
    filtered_img1 = f(img_1)
    pylab.subplot(1,3,1);pylab.axis('off');pylab.imshow(img)
    pylab.gray()
    pylab.subplot(1,3,2);pylab.axis('off');pylab.imshow(filtered_img[0,0,:,:])
    pylab.subplot(1, 3, 3);pylab.axis('off');pylab.imshow(filtered_img[0, 1, :, :])
    pylab.show()
if __name__ == '__main__':
    testConv2d()
