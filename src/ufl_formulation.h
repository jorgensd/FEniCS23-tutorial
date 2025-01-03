
// This code conforms with the UFC specification version 2018.2.0.dev0
// and was automatically generated by FFCx version 0.8.0.
//
// This code was generated with the following options:
//
//  {'epsilon': 1e-14,
//   'output_directory': '/root/shared/src',
//   'profile': False,
//   'scalar_type': 'float64',
//   'sum_factorization': False,
//   'table_atol': 1e-09,
//   'table_rtol': 1e-06,
//   'ufl_file': ['/root/shared/src/ufl_formulation.py'],
//   'verbosity': 30,
//   'visualise': True}

#pragma once
#include <ufcx.h>

#ifdef __cplusplus
extern "C" {
#endif

extern ufcx_finite_element element_74d6cdfde3ea4b5d423c7bf4310320a0acafddfe;

extern ufcx_finite_element element_e60371328a49148d7039c103115050855ae79a18;

extern ufcx_finite_element element_9c6cd84bfc267751ad970123baa9b98e3391e3e7;

extern ufcx_finite_element element_f4f67755fd7baf4a3b706b45903ee041f79a5f4e;

extern ufcx_dofmap dofmap_74d6cdfde3ea4b5d423c7bf4310320a0acafddfe;

extern ufcx_dofmap dofmap_e60371328a49148d7039c103115050855ae79a18;

extern ufcx_dofmap dofmap_9c6cd84bfc267751ad970123baa9b98e3391e3e7;

extern ufcx_dofmap dofmap_f4f67755fd7baf4a3b706b45903ee041f79a5f4e;

extern ufcx_integral integral_444ae3de485a81d987c5e8c21247536bf812e211;

extern ufcx_integral integral_c8c7cee93b610bf40fb284d1c05d7b02fd51f3b6;

extern ufcx_form form_90f642a9a6a793322d73d559b0f550b853e94484;

// Helper used to create form using name which was given to the
// form in the UFL file.
// This helper is called in user c++ code.
//
extern ufcx_form* form_ufl_formulation_a;

// Helper used to create function space using function name
// i.e. name of the Python variable.
//
ufcx_function_space* functionspace_form_ufl_formulation_a(const char* function_name);

extern ufcx_form form_62c2cfe51a2ce925c4a5356b7dd767fc07b70c6d;

// Helper used to create form using name which was given to the
// form in the UFL file.
// This helper is called in user c++ code.
//
extern ufcx_form* form_ufl_formulation_L;

// Helper used to create function space using function name
// i.e. name of the Python variable.
//
ufcx_function_space* functionspace_form_ufl_formulation_L(const char* function_name);

#ifdef __cplusplus
}
#endif
