#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <scalarArray.h>
#include <configurators.h>
#include <registration.h>

namespace py = pybind11;

template <typename DataType>
class NumPy2DScalarArray : public qc::ScalarArray<DataType, qc::QC_2D> {
public:
    explicit NumPy2DScalarArray ( py::array_t<DataType, py::array::c_style> array )
      : qc::ScalarArray<DataType, qc::QC_2D> ( array.shape(1), array.shape(0), const_cast<double*>(array.data()) )
    {
        if ( array.ndim() != 2 )
            throw aol::Exception ( "Expected 2D NumPy array!\n", __FILE__, __LINE__ );
    }
};


template <typename DataType>
class NumPy2DMultiArray : public qc::MultiArray<DataType, qc::QC_2D> {
public:
    explicit NumPy2DMultiArray ( py::array_t<DataType, py::array::c_style> array )
      : qc::MultiArray<DataType, qc::QC_2D> ( array.shape(2), array.shape(1), qc::MultiArray<DataType, qc::QC_2D>::CREATE_NO_ARRAYS )
    {
        if ( array.ndim() != 3 )
            throw aol::Exception ( "Expected 3D NumPy array!\n", __FILE__, __LINE__ );

        if ( array.shape(0) != 2 )
            throw aol::Exception ( "Expected shape(0) to be 2!\n", __FILE__, __LINE__ );

        const int numX = array.shape(2);
        const int numY = array.shape(1);

        for ( int i = 0; i < 2; ++i ) {
            this->appendReference ( * ( new qc::ScalarArray<DataType, qc::QC_2D> ( numX, numY, const_cast<double*>(&(array.data()[i * numX * numY])), aol::FLAT_COPY ) ), true );
        }
    }
};


void array_test(py::array_t<double, py::array::c_style> array){
    std::cerr << "Hello world\n";
    std::cerr << array.itemsize();
    const int dim = array.ndim();
    for (int i = 0; i < dim; ++i)
        std::cerr << array.shape(i);
    double *data = const_cast<double*>(array.data());
    for (int i = 0; i < array.size(); ++i)
        std::cerr << data[i] << std::endl;

    NumPy2DScalarArray<double> qcArray(array);

    std::cerr << qcArray << std::endl;
    qcArray[0, 0] = 666.;
    std::cerr << qcArray << std::endl;
    std::cerr << qcArray.norm() << std::endl;
}


py::array_t<double> register_images(py::array_t<double, py::array::c_style> Reference, py::array_t<double, py::array::c_style> Template, py::array_t<double, py::array::c_style> Deformation){
  try {
    NumPy2DScalarArray<double> referenceArray(Reference);
    NumPy2DScalarArray<double> templateArray(Template);
    NumPy2DMultiArray<double> deformationArray(Deformation);

    qc::GridSize<qc::QC_2D> gridSize ( referenceArray );
    typedef qc::RectangularGridConfigurator<double, qc::QC_2D, aol::GaussQuadrature<double, qc::QC_2D, 3> > ConfType;
    const ConfType::InitType grid ( gridSize );
    const double lambda = 10;

    qc::NCCRegistrationConfigurator<ConfType> regisConf;
    qc::DirichletRegularizationConfigurator<ConfType> regulConf ( grid );
    qc::StandardRegistration<ConfType, ConfType::ArrayType, qc::NCCRegistrationConfigurator<ConfType>, qc::DirichletRegularizationConfigurator<ConfType> >
      stdRegistration ( referenceArray, templateArray, regisConf, regulConf, lambda );

    stdRegistration.findTransformation ( deformationArray );

    qc::ScalarArray<double, qc::QC_2D> match ( grid );
    qc::DeformImage<ConfType> ( templateArray, grid, match, deformationArray );


    return py::array_t<double>(
            {match.getNumY(), match.getNumX()}, // shape
            {match.getNumX()*8, 8}, // C-style contiguous strides for double
            match.getData()); // the data pointer
  }
  catch ( aol::Exception &el ) {
    el.dump();
  }
  return py::array_t<double>();
}


PYBIND11_MODULE(cmake_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           array_test
           register_images
    )pbdoc";

    m.def("array_test", &array_test, "Hello World numpy array manipulation example");
    m.def("register_images", &register_images, "A function which registers two numbers",
          py::arg("reference_image"), py::arg("template_image"), py::arg("displacement"));

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
