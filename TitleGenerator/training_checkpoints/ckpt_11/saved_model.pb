��"
�!� 
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
�
StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628�!
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
t
gru/VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
d*
shared_namegru/Variable
m
 gru/Variable/Read/ReadVariableOpReadVariableOpgru/Variable*
_output_shapes

:
d*
dtype0
z
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:L*"
shared_nameAdam/v/dense/bias
s
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes
:L*
dtype0
z
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:L*"
shared_nameAdam/m/dense/bias
s
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes
:L*
dtype0
�
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dL*$
shared_nameAdam/v/dense/kernel
{
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes

:dL*
dtype0
�
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dL*$
shared_nameAdam/m/dense/kernel
{
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes

:dL*
dtype0
�
Adam/v/gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameAdam/v/gru/gru_cell/bias
�
,Adam/v/gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpAdam/v/gru/gru_cell/bias*
_output_shapes
:	�*
dtype0
�
Adam/m/gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameAdam/m/gru/gru_cell/bias
�
,Adam/m/gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpAdam/m/gru/gru_cell/bias*
_output_shapes
:	�*
dtype0
�
$Adam/v/gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*5
shared_name&$Adam/v/gru/gru_cell/recurrent_kernel
�
8Adam/v/gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp$Adam/v/gru/gru_cell/recurrent_kernel*
_output_shapes
:	d�*
dtype0
�
$Adam/m/gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*5
shared_name&$Adam/m/gru/gru_cell/recurrent_kernel
�
8Adam/m/gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp$Adam/m/gru/gru_cell/recurrent_kernel*
_output_shapes
:	d�*
dtype0
�
Adam/v/gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*+
shared_nameAdam/v/gru/gru_cell/kernel
�
.Adam/v/gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/v/gru/gru_cell/kernel*
_output_shapes
:	d�*
dtype0
�
Adam/m/gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*+
shared_nameAdam/m/gru/gru_cell/kernel
�
.Adam/m/gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/m/gru/gru_cell/kernel*
_output_shapes
:	d�*
dtype0
�
Adam/v/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Ld*,
shared_nameAdam/v/embedding/embeddings
�
/Adam/v/embedding/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding/embeddings*
_output_shapes

:Ld*
dtype0
�
Adam/m/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Ld*,
shared_nameAdam/m/embedding/embeddings
�
/Adam/m/embedding/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding/embeddings*
_output_shapes

:Ld*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	

gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*"
shared_namegru/gru_cell/bias
x
%gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell/bias*
_output_shapes
:	�*
dtype0
�
gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*.
shared_namegru/gru_cell/recurrent_kernel
�
1gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/recurrent_kernel*
_output_shapes
:	d�*
dtype0
�
gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*$
shared_namegru/gru_cell/kernel
|
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel*
_output_shapes
:	d�*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:L*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:L*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dL*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:dL*
dtype0
�
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Ld*%
shared_nameembedding/embeddings
}
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes

:Ld*
dtype0
�
serving_default_embedding_inputPlaceholder*'
_output_shapes
:
���������*
dtype0*
shape:
���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_embedding_inputembedding/embeddingsgru/Variablegru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biasdense/kernel
dense/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:
���������L*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_2375410

NoOpNoOp
�/
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�.
value�.B�. B�.
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
.
0
%1
&2
'3
#4
$5*
.
0
%1
&2
'3
#4
$5*
* 
�
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*

-trace_0
.trace_1* 

/trace_0
0trace_1* 
* 
�
1
_variables
2_iterations
3_learning_rate
4_index_dict
5
_momentums
6_velocities
7_update_step_xla*

8serving_default* 

0*

0*
* 
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

>trace_0* 

?trace_0* 
hb
VARIABLE_VALUEembedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

%0
&1
'2*

%0
&1
'2*
* 
�

@states
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ftrace_0
Gtrace_1
Htrace_2
Itrace_3* 
6
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_3* 
* 
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T_random_generator

%kernel
&recurrent_kernel
'bias*
* 

#0
$1*

#0
$1*
* 
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

Ztrace_0* 

[trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEgru/gru_cell/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEgru/gru_cell/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEgru/gru_cell/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

\0*
* 
* 
* 
* 
* 
* 
b
20
]1
^2
_3
`4
a5
b6
c7
d8
e9
f10
g11
h12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
]0
_1
a2
c3
e4
g5*
.
^0
`1
b2
d3
f4
h5*
* 
* 
* 
* 
* 
* 
* 
* 
* 

i0*
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

%0
&1
'2*

%0
&1
'2*
* 
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
o	variables
p	keras_api
	qtotal
	rcount*
f`
VARIABLE_VALUEAdam/m/embedding/embeddings1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/embedding/embeddings1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/gru/gru_cell/kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/gru/gru_cell/kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/m/gru/gru_cell/recurrent_kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/v/gru/gru_cell/recurrent_kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/gru/gru_cell/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/gru/gru_cell/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/dense/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/dense/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEgru/VariableBlayer_with_weights-1/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 

q0
r1*

o	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameembedding/embeddingsdense/kernel
dense/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/bias	iterationlearning_rateAdam/m/embedding/embeddingsAdam/v/embedding/embeddingsAdam/m/gru/gru_cell/kernelAdam/v/gru/gru_cell/kernel$Adam/m/gru/gru_cell/recurrent_kernel$Adam/v/gru/gru_cell/recurrent_kernelAdam/m/gru/gru_cell/biasAdam/v/gru/gru_cell/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasgru/VariabletotalcountConst*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_2377153
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingsdense/kernel
dense/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/bias	iterationlearning_rateAdam/m/embedding/embeddingsAdam/v/embedding/embeddingsAdam/m/gru/gru_cell/kernelAdam/v/gru/gru_cell/kernel$Adam/m/gru/gru_cell/recurrent_kernel$Adam/v/gru/gru_cell/recurrent_kernelAdam/m/gru/gru_cell/biasAdam/v/gru/gru_cell/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasgru/Variabletotalcount*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_2377231�� 
�
�
'__inference_dense_layer_call_fn_2376963

inputs
unknown:dL
	unknown_0:L
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:
���������L*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_2374911s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:
���������L<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:
���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2376959:'#
!
_user_specified_name	2376957:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�;
�
 __inference_standard_gru_2374249

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:�:�*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_mask\
MatMulMatMulstrided_slice_1:output:0kernel*
T0*
_output_shapes
:	
�`
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	
�Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitV
MatMul_1MatMulinit_hrecurrent_kernel*
T0*
_output_shapes
:	
�d
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	
�S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitW
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes

:
dD
SigmoidSigmoidadd:z:0*
T0*
_output_shapes

:
dY
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes

:
dH
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes

:
dT
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes

:
dP
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes

:
d@
TanhTanh	add_2:z:0*
T0*
_output_shapes

:
dJ
mul_1MulSigmoid:y:0init_h*
T0*
_output_shapes

:
dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes

:
dH
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes

:
dM
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes

:
dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :
d: : :	d�:�:	d�:�* 
_read_only_resource_inputs
 *
bodyR
while_body_2374160*
condR
while_cond_2374159*M
output_shapes<
:: : : : :
d: : :	d�:�:	d�:�*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������
d*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:
���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?W
IdentityIdentitystrided_slice_2:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:
���������dO

Identity_2Identitywhile:output:4*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_1d4607a8-8de1-46eb-965d-c935dbf59148*
api_preferred_deviceCPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�

�
while_cond_2374988
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2374988___redundant_placeholder05
1while_while_cond_2374988___redundant_placeholder15
1while_while_cond_2374988___redundant_placeholder25
1while_while_cond_2374988___redundant_placeholder35
1while_while_cond_2374988___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: : : : :
d: ::::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
�
%__inference_signature_wrapper_2375410
embedding_input
unknown:Ld
	unknown_0:
d
	unknown_1:	d�
	unknown_2:	d�
	unknown_3:	�
	unknown_4:dL
	unknown_5:L
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:
���������L*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_2373725s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:
���������L<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:
���������: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2375406:'#
!
_user_specified_name	2375404:'#
!
_user_specified_name	2375402:'#
!
_user_specified_name	2375400:'#
!
_user_specified_name	2375398:'#
!
_user_specified_name	2375396:'#
!
_user_specified_name	2375394:X T
'
_output_shapes
:
���������
)
_user_specified_nameembedding_input
�
�
B__inference_dense_layer_call_and_return_conditional_losses_2374911

inputs3
!tensordot_readvariableop_resource:dL-
biasadd_readvariableop_resource:L
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:dL*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:
���������d�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������L[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:LY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:
���������Lr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:L*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:
���������Lc
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:
���������LV
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:
���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�-
�
while_body_2375913
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
d*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*
_output_shapes
:	
�s
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*
_output_shapes
:	
�W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitz
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*
_output_shapes
:	
�y
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*
_output_shapes
:	
�Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_spliti
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*
_output_shapes

:
dP
while/SigmoidSigmoidwhile/add:z:0*
T0*
_output_shapes

:
dk
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*
_output_shapes

:
dT
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*
_output_shapes

:
df
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*
_output_shapes

:
db
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*
_output_shapes

:
dL

while/TanhTanhwhile/add_2:z:0*
T0*
_output_shapes

:
dc
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes

:
dP
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*
_output_shapes

:
dZ
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*
_output_shapes

:
d_
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*
_output_shapes

:
d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: V
while/Identity_4Identitywhile/add_3:z:0*
T0*
_output_shapes

:
d"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :
d: : :	d�:�:	d�:�:D
@

_output_shapes	
:�
!
_user_specified_name	unstack:Q	M

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:D@

_output_shapes	
:�
!
_user_specified_name	unstack:GC

_output_shapes
:	d�
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�	
�
%__inference_gru_layer_call_fn_2375452
inputs_0
unknown:
d
	unknown_0:	d�
	unknown_1:	d�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:
���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_gru_layer_call_and_return_conditional_losses_2374463s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:
���������d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:
���������d: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2375448:'#
!
_user_specified_name	2375446:'#
!
_user_specified_name	2375444:'#
!
_user_specified_name	2375442:U Q
+
_output_shapes
:
���������d
"
_user_specified_name
inputs_0
�
�
@__inference_gru_layer_call_and_return_conditional_losses_2374872

inputs.
read_readvariableop_resource:
d1
read_1_readvariableop_resource:	d�1
read_2_readvariableop_resource:	d�1
read_3_readvariableop_resource:	�

identity_4��AssignVariableOp�Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp�Read_3/ReadVariableOpp
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:
d*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:
du
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	d�*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�u
Read_3/ReadVariableOpReadVariableOpread_3_readvariableop_resource*
_output_shapes
:	�*
dtype0_

Identity_3IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
PartitionedCallPartitionedCallinputsIdentity:output:0Identity_1:output:0Identity_2:output:0Identity_3:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:
d:
���������d:
d: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2374658�
AssignVariableOpAssignVariableOpread_readvariableop_resourcePartitionedCall:output:2^Read/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(m

Identity_4IdentityPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:
���������d�
NoOpNoOp^AssignVariableOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp^Read_3/ReadVariableOp*
_output_shapes
 "!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:
���������d: : : : 2$
AssignVariableOpAssignVariableOp2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp2.
Read_3/ReadVariableOpRead_3/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�

�
while_cond_2374568
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2374568___redundant_placeholder05
1while_while_cond_2374568___redundant_placeholder15
1while_while_cond_2374568___redundant_placeholder25
1while_while_cond_2374568___redundant_placeholder35
1while_while_cond_2374568___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: : : : :
d: ::::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
�
@__inference_gru_layer_call_and_return_conditional_losses_2374094

inputs.
read_readvariableop_resource:
d1
read_1_readvariableop_resource:	d�1
read_2_readvariableop_resource:	d�1
read_3_readvariableop_resource:	�

identity_4��AssignVariableOp�Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp�Read_3/ReadVariableOpp
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:
d*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:
du
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	d�*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�u
Read_3/ReadVariableOpReadVariableOpread_3_readvariableop_resource*
_output_shapes
:	�*
dtype0_

Identity_3IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
PartitionedCallPartitionedCallinputsIdentity:output:0Identity_1:output:0Identity_2:output:0Identity_3:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:
d:
���������d:
d: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2373880�
AssignVariableOpAssignVariableOpread_readvariableop_resourcePartitionedCall:output:2^Read/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(m

Identity_4IdentityPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:
���������d�
NoOpNoOp^AssignVariableOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp^Read_3/ReadVariableOp*
_output_shapes
 "!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:
���������d: : : : 2$
AssignVariableOpAssignVariableOp2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp2.
Read_3/ReadVariableOpRead_3/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�=
�
'__forward_gpu_gru_with_fallback_2375844

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1
	transpose

expanddims
cudnnrnn_input_c

concat

cudnnrnn_2
transpose_perm
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "#
concat_axisconcat/axis:output:0"
concatconcat_0:output:0"!

cudnnrnn_0CudnnRNN:output_c:0"&

cudnnrnn_1CudnnRNN:reserve_space:0"!

cudnnrnn_2CudnnRNN:output_h:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"
cudnnrnnCudnnRNN:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_3378f1f8-d8c8-407b-8966-68b8acf9ba31*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2375710_2375845*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�

�
while_cond_2373395
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2373395___redundant_placeholder05
1while_while_cond_2373395___redundant_placeholder15
1while_while_cond_2373395___redundant_placeholder25
1while_while_cond_2373395___redundant_placeholder35
1while_while_cond_2373395___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: : : : :
d: ::::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�;
�
 __inference_standard_gru_2376002

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:�:�*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_mask\
MatMulMatMulstrided_slice_1:output:0kernel*
T0*
_output_shapes
:	
�`
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	
�Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitV
MatMul_1MatMulinit_hrecurrent_kernel*
T0*
_output_shapes
:	
�d
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	
�S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitW
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes

:
dD
SigmoidSigmoidadd:z:0*
T0*
_output_shapes

:
dY
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes

:
dH
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes

:
dT
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes

:
dP
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes

:
d@
TanhTanh	add_2:z:0*
T0*
_output_shapes

:
dJ
mul_1MulSigmoid:y:0init_h*
T0*
_output_shapes

:
dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes

:
dH
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes

:
dM
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes

:
dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :
d: : :	d�:�:	d�:�* 
_read_only_resource_inputs
 *
bodyR
while_body_2375913*
condR
while_cond_2375912*M
output_shapes<
:: : : : :
d: : :	d�:�:	d�:�*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������
d*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:
���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?W
IdentityIdentitystrided_slice_2:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:
���������dO

Identity_2Identitywhile:output:4*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_147464ad-d9c1-4bc4-af9c-aa73203e3343*
api_preferred_deviceCPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�m
�
#__inference__traced_restore_2377231
file_prefix7
%assignvariableop_embedding_embeddings:Ld1
assignvariableop_1_dense_kernel:dL+
assignvariableop_2_dense_bias:L9
&assignvariableop_3_gru_gru_cell_kernel:	d�C
0assignvariableop_4_gru_gru_cell_recurrent_kernel:	d�7
$assignvariableop_5_gru_gru_cell_bias:	�&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: @
.assignvariableop_8_adam_m_embedding_embeddings:Ld@
.assignvariableop_9_adam_v_embedding_embeddings:LdA
.assignvariableop_10_adam_m_gru_gru_cell_kernel:	d�A
.assignvariableop_11_adam_v_gru_gru_cell_kernel:	d�K
8assignvariableop_12_adam_m_gru_gru_cell_recurrent_kernel:	d�K
8assignvariableop_13_adam_v_gru_gru_cell_recurrent_kernel:	d�?
,assignvariableop_14_adam_m_gru_gru_cell_bias:	�?
,assignvariableop_15_adam_v_gru_gru_cell_bias:	�9
'assignvariableop_16_adam_m_dense_kernel:dL9
'assignvariableop_17_adam_v_dense_kernel:dL3
%assignvariableop_18_adam_m_dense_bias:L3
%assignvariableop_19_adam_v_dense_bias:L2
 assignvariableop_20_gru_variable:
d#
assignvariableop_21_total: #
assignvariableop_22_count: 
identity_24��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-1/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*t
_output_shapesb
`::::::::::::::::::::::::*&
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_kernelIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_biasIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp&assignvariableop_3_gru_gru_cell_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp0assignvariableop_4_gru_gru_cell_recurrent_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp$assignvariableop_5_gru_gru_cell_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_adam_m_embedding_embeddingsIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_adam_v_embedding_embeddingsIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp.assignvariableop_10_adam_m_gru_gru_cell_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp.assignvariableop_11_adam_v_gru_gru_cell_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp8assignvariableop_12_adam_m_gru_gru_cell_recurrent_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp8assignvariableop_13_adam_v_gru_gru_cell_recurrent_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp,assignvariableop_14_adam_m_gru_gru_cell_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_v_gru_gru_cell_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_m_dense_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_v_dense_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_m_dense_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp%assignvariableop_19_adam_v_dense_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp assignvariableop_20_gru_variableIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_24IdentityIdentity_23:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_24Identity_24:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%!

_user_specified_namecount:%!

_user_specified_nametotal:,(
&
_user_specified_namegru/Variable:1-
+
_user_specified_nameAdam/v/dense/bias:1-
+
_user_specified_nameAdam/m/dense/bias:3/
-
_user_specified_nameAdam/v/dense/kernel:3/
-
_user_specified_nameAdam/m/dense/kernel:84
2
_user_specified_nameAdam/v/gru/gru_cell/bias:84
2
_user_specified_nameAdam/m/gru/gru_cell/bias:D@
>
_user_specified_name&$Adam/v/gru/gru_cell/recurrent_kernel:D@
>
_user_specified_name&$Adam/m/gru/gru_cell/recurrent_kernel::6
4
_user_specified_nameAdam/v/gru/gru_cell/kernel::6
4
_user_specified_nameAdam/m/gru/gru_cell/kernel:;
7
5
_user_specified_nameAdam/v/embedding/embeddings:;	7
5
_user_specified_nameAdam/m/embedding/embeddings:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:1-
+
_user_specified_namegru/gru_cell/bias:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:40
.
_user_specified_nameembedding/embeddings:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
while_cond_2373790
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2373790___redundant_placeholder05
1while_while_cond_2373790___redundant_placeholder15
1while_while_cond_2373790___redundant_placeholder25
1while_while_cond_2373790___redundant_placeholder35
1while_while_cond_2373790___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: : : : :
d: ::::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�-
�
while_body_2374160
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
d*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*
_output_shapes
:	
�s
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*
_output_shapes
:	
�W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitz
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*
_output_shapes
:	
�y
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*
_output_shapes
:	
�Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_spliti
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*
_output_shapes

:
dP
while/SigmoidSigmoidwhile/add:z:0*
T0*
_output_shapes

:
dk
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*
_output_shapes

:
dT
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*
_output_shapes

:
df
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*
_output_shapes

:
db
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*
_output_shapes

:
dL

while/TanhTanhwhile/add_2:z:0*
T0*
_output_shapes

:
dc
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes

:
dP
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*
_output_shapes

:
dZ
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*
_output_shapes

:
d_
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*
_output_shapes

:
d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: V
while/Identity_4Identitywhile/add_3:z:0*
T0*
_output_shapes

:
d"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :
d: : :	d�:�:	d�:�:D
@

_output_shapes	
:�
!
_user_specified_name	unstack:Q	M

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:D@

_output_shapes	
:�
!
_user_specified_name	unstack:GC

_output_shapes
:	d�
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�4
�
)__inference_gpu_gru_with_fallback_2376447

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:��U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_d0df98bb-a0f8-4a1d-a383-06362d752882*
api_preferred_deviceGPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
��
�

<__inference___backward_gpu_gru_with_fallback_2376817_2376952
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat5
1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn=
9gradients_transpose_grad_invertpermutation_transpose_perm)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�U
gradients/grad_ys_0Identityplaceholder*
T0*
_output_shapes

:
dd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:
���������dW
gradients/grad_ys_2Identityplaceholder_2*
T0*
_output_shapes

:
dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::���
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:���������
d*
shrink_axis_mask�
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:�
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������
dq
gradients/Squeeze_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
   d   �
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*"
_output_shapes
:
d�
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:���������
da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*C
_output_shapes1
/:���������
d:
d: :��*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:
���������dp
gradients/ExpandDims_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*
_output_shapes

:
d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�Ng
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:d�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:d�
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
gradients/split_2_grad/concatConcatV2)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	d��
gradients/split_1_grad/concatConcatV2(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d�m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ,  �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	�r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:
���������dk

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*
_output_shapes

:
df

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	d�h

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	d�i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	�"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:
d:
���������d:
d: :���������
d:: ::���������
d:
d: :��:
d:: ::::::: : : *<
api_implements*(gru_7393682a-ed09-4efb-89b0-ce0fc0b05b91*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2376951*
go_backwards( *

time_major( :IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:LH
"
_output_shapes
:
d
"
_user_specified_name
CudnnRNN:D@

_output_shapes

:��
 
_user_specified_nameconcat:H
D

_output_shapes
: 
*
_user_specified_nameCudnnRNN/input_c:N	J
"
_output_shapes
:
d
$
_user_specified_name
ExpandDims:VR
+
_output_shapes
:���������
d
#
_user_specified_name	transpose:B>

_output_shapes
:
"
_user_specified_name
CudnnRNN:@<

_output_shapes
: 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:UQ
+
_output_shapes
:���������
d
"
_user_specified_name
CudnnRNN:

_output_shapes
: :$ 

_output_shapes

:
d:1-
+
_output_shapes
:
���������d:$  

_output_shapes

:
d
��
�

<__inference___backward_gpu_gru_with_fallback_2376448_2376583
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat5
1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn=
9gradients_transpose_grad_invertpermutation_transpose_perm)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�U
gradients/grad_ys_0Identityplaceholder*
T0*
_output_shapes

:
dd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:
���������dW
gradients/grad_ys_2Identityplaceholder_2*
T0*
_output_shapes

:
dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::���
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:���������
d*
shrink_axis_mask�
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:�
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������
dq
gradients/Squeeze_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
   d   �
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*"
_output_shapes
:
d�
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:���������
da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*C
_output_shapes1
/:���������
d:
d: :��*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:
���������dp
gradients/ExpandDims_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*
_output_shapes

:
d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�Ng
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:d�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:d�
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
gradients/split_2_grad/concatConcatV2)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	d��
gradients/split_1_grad/concatConcatV2(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d�m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ,  �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	�r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:
���������dk

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*
_output_shapes

:
df

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	d�h

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	d�i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	�"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:
d:
���������d:
d: :���������
d:: ::���������
d:
d: :��:
d:: ::::::: : : *<
api_implements*(gru_d0df98bb-a0f8-4a1d-a383-06362d752882*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2376582*
go_backwards( *

time_major( :IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:LH
"
_output_shapes
:
d
"
_user_specified_name
CudnnRNN:D@

_output_shapes

:��
 
_user_specified_nameconcat:H
D

_output_shapes
: 
*
_user_specified_nameCudnnRNN/input_c:N	J
"
_output_shapes
:
d
$
_user_specified_name
ExpandDims:VR
+
_output_shapes
:���������
d
#
_user_specified_name	transpose:B>

_output_shapes
:
"
_user_specified_name
CudnnRNN:@<

_output_shapes
: 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:UQ
+
_output_shapes
:���������
d
"
_user_specified_name
CudnnRNN:

_output_shapes
: :$ 

_output_shapes

:
d:1-
+
_output_shapes
:
���������d:$  

_output_shapes

:
d
��
�

<__inference___backward_gpu_gru_with_fallback_2374326_2374461
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat5
1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn=
9gradients_transpose_grad_invertpermutation_transpose_perm)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�U
gradients/grad_ys_0Identityplaceholder*
T0*
_output_shapes

:
dd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:
���������dW
gradients/grad_ys_2Identityplaceholder_2*
T0*
_output_shapes

:
dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::���
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:���������
d*
shrink_axis_mask�
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:�
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������
dq
gradients/Squeeze_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
   d   �
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*"
_output_shapes
:
d�
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:���������
da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*C
_output_shapes1
/:���������
d:
d: :��*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:
���������dp
gradients/ExpandDims_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*
_output_shapes

:
d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�Ng
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:d�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:d�
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
gradients/split_2_grad/concatConcatV2)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	d��
gradients/split_1_grad/concatConcatV2(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d�m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ,  �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	�r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:
���������dk

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*
_output_shapes

:
df

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	d�h

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	d�i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	�"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:
d:
���������d:
d: :���������
d:: ::���������
d:
d: :��:
d:: ::::::: : : *<
api_implements*(gru_1d4607a8-8de1-46eb-965d-c935dbf59148*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2374460*
go_backwards( *

time_major( :IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:LH
"
_output_shapes
:
d
"
_user_specified_name
CudnnRNN:D@

_output_shapes

:��
 
_user_specified_nameconcat:H
D

_output_shapes
: 
*
_user_specified_nameCudnnRNN/input_c:N	J
"
_output_shapes
:
d
$
_user_specified_name
ExpandDims:VR
+
_output_shapes
:���������
d
#
_user_specified_name	transpose:B>

_output_shapes
:
"
_user_specified_name
CudnnRNN:@<

_output_shapes
: 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:UQ
+
_output_shapes
:���������
d
"
_user_specified_name
CudnnRNN:

_output_shapes
: :$ 

_output_shapes

:
d:1-
+
_output_shapes
:
���������d:$  

_output_shapes

:
d
�;
�
 __inference_standard_gru_2373880

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:�:�*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_mask\
MatMulMatMulstrided_slice_1:output:0kernel*
T0*
_output_shapes
:	
�`
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	
�Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitV
MatMul_1MatMulinit_hrecurrent_kernel*
T0*
_output_shapes
:	
�d
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	
�S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitW
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes

:
dD
SigmoidSigmoidadd:z:0*
T0*
_output_shapes

:
dY
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes

:
dH
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes

:
dT
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes

:
dP
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes

:
d@
TanhTanh	add_2:z:0*
T0*
_output_shapes

:
dJ
mul_1MulSigmoid:y:0init_h*
T0*
_output_shapes

:
dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes

:
dH
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes

:
dM
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes

:
dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :
d: : :	d�:�:	d�:�* 
_read_only_resource_inputs
 *
bodyR
while_body_2373791*
condR
while_cond_2373790*M
output_shapes<
:: : : : :
d: : :	d�:�:	d�:�*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������
d*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:
���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?W
IdentityIdentitystrided_slice_2:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:
���������dO

Identity_2Identitywhile:output:4*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_0660cc53-f4b6-450f-8c94-e84ee22a46c0*
api_preferred_deviceCPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�
�
@__inference_gru_layer_call_and_return_conditional_losses_2376585

inputs.
read_readvariableop_resource:
d1
read_1_readvariableop_resource:	d�1
read_2_readvariableop_resource:	d�1
read_3_readvariableop_resource:	�

identity_4��AssignVariableOp�Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp�Read_3/ReadVariableOpp
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:
d*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:
du
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	d�*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�u
Read_3/ReadVariableOpReadVariableOpread_3_readvariableop_resource*
_output_shapes
:	�*
dtype0_

Identity_3IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
PartitionedCallPartitionedCallinputsIdentity:output:0Identity_1:output:0Identity_2:output:0Identity_3:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:
d:
���������d:
d: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2376371�
AssignVariableOpAssignVariableOpread_readvariableop_resourcePartitionedCall:output:2^Read/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(m

Identity_4IdentityPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:
���������d�
NoOpNoOp^AssignVariableOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp^Read_3/ReadVariableOp*
_output_shapes
 "!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:
���������d: : : : 2$
AssignVariableOpAssignVariableOp2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp2.
Read_3/ReadVariableOpRead_3/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�	
�
%__inference_gru_layer_call_fn_2375439
inputs_0
unknown:
d
	unknown_0:	d�
	unknown_1:	d�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:
���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_gru_layer_call_and_return_conditional_losses_2374094s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:
���������d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:
���������d: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2375435:'#
!
_user_specified_name	2375433:'#
!
_user_specified_name	2375431:'#
!
_user_specified_name	2375429:U Q
+
_output_shapes
:
���������d
"
_user_specified_name
inputs_0
�;
�
 __inference_standard_gru_2375078

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:�:�*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_mask\
MatMulMatMulstrided_slice_1:output:0kernel*
T0*
_output_shapes
:	
�`
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	
�Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitV
MatMul_1MatMulinit_hrecurrent_kernel*
T0*
_output_shapes
:	
�d
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	
�S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitW
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes

:
dD
SigmoidSigmoidadd:z:0*
T0*
_output_shapes

:
dY
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes

:
dH
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes

:
dT
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes

:
dP
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes

:
d@
TanhTanh	add_2:z:0*
T0*
_output_shapes

:
dJ
mul_1MulSigmoid:y:0init_h*
T0*
_output_shapes

:
dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes

:
dH
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes

:
dM
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes

:
dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :
d: : :	d�:�:	d�:�* 
_read_only_resource_inputs
 *
bodyR
while_body_2374989*
condR
while_cond_2374988*M
output_shapes<
:: : : : :
d: : :	d�:�:	d�:�*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������
d*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:
���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?W
IdentityIdentitystrided_slice_2:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:
���������dO

Identity_2Identitywhile:output:4*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_999d41c9-6a05-4f02-bc6c-cc5bbb6e46d4*
api_preferred_deviceCPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�-
�
while_body_2376651
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
d*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*
_output_shapes
:	
�s
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*
_output_shapes
:	
�W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitz
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*
_output_shapes
:	
�y
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*
_output_shapes
:	
�Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_spliti
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*
_output_shapes

:
dP
while/SigmoidSigmoidwhile/add:z:0*
T0*
_output_shapes

:
dk
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*
_output_shapes

:
dT
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*
_output_shapes

:
df
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*
_output_shapes

:
db
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*
_output_shapes

:
dL

while/TanhTanhwhile/add_2:z:0*
T0*
_output_shapes

:
dc
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes

:
dP
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*
_output_shapes

:
dZ
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*
_output_shapes

:
d_
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*
_output_shapes

:
d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: V
while/Identity_4Identitywhile/add_3:z:0*
T0*
_output_shapes

:
d"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :
d: : :	d�:�:	d�:�:D
@

_output_shapes	
:�
!
_user_specified_name	unstack:Q	M

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:D@

_output_shapes	
:�
!
_user_specified_name	unstack:GC

_output_shapes
:	d�
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
��
�

<__inference___backward_gpu_gru_with_fallback_2375710_2375845
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat5
1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn=
9gradients_transpose_grad_invertpermutation_transpose_perm)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�U
gradients/grad_ys_0Identityplaceholder*
T0*
_output_shapes

:
dd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:
���������dW
gradients/grad_ys_2Identityplaceholder_2*
T0*
_output_shapes

:
dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::���
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:���������
d*
shrink_axis_mask�
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:�
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������
dq
gradients/Squeeze_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
   d   �
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*"
_output_shapes
:
d�
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:���������
da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*C
_output_shapes1
/:���������
d:
d: :��*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:
���������dp
gradients/ExpandDims_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*
_output_shapes

:
d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�Ng
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:d�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:d�
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
gradients/split_2_grad/concatConcatV2)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	d��
gradients/split_1_grad/concatConcatV2(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d�m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ,  �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	�r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:
���������dk

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*
_output_shapes

:
df

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	d�h

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	d�i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	�"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:
d:
���������d:
d: :���������
d:: ::���������
d:
d: :��:
d:: ::::::: : : *<
api_implements*(gru_3378f1f8-d8c8-407b-8966-68b8acf9ba31*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2375844*
go_backwards( *

time_major( :IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:LH
"
_output_shapes
:
d
"
_user_specified_name
CudnnRNN:D@

_output_shapes

:��
 
_user_specified_nameconcat:H
D

_output_shapes
: 
*
_user_specified_nameCudnnRNN/input_c:N	J
"
_output_shapes
:
d
$
_user_specified_name
ExpandDims:VR
+
_output_shapes
:���������
d
#
_user_specified_name	transpose:B>

_output_shapes
:
"
_user_specified_name
CudnnRNN:@<

_output_shapes
: 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:UQ
+
_output_shapes
:���������
d
"
_user_specified_name
CudnnRNN:

_output_shapes
: :$ 

_output_shapes

:
d:1-
+
_output_shapes
:
���������d:$  

_output_shapes

:
d
�4
�
)__inference_gpu_gru_with_fallback_2376078

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:��U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_147464ad-d9c1-4bc4-af9c-aa73203e3343*
api_preferred_deviceGPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�4
�
)__inference_gpu_gru_with_fallback_2375709

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:��U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_3378f1f8-d8c8-407b-8966-68b8acf9ba31*
api_preferred_deviceGPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�

�
while_cond_2374159
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2374159___redundant_placeholder05
1while_while_cond_2374159___redundant_placeholder15
1while_while_cond_2374159___redundant_placeholder25
1while_while_cond_2374159___redundant_placeholder35
1while_while_cond_2374159___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: : : : :
d: ::::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�-
�
while_body_2374989
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
d*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*
_output_shapes
:	
�s
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*
_output_shapes
:	
�W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitz
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*
_output_shapes
:	
�y
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*
_output_shapes
:	
�Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_spliti
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*
_output_shapes

:
dP
while/SigmoidSigmoidwhile/add:z:0*
T0*
_output_shapes

:
dk
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*
_output_shapes

:
dT
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*
_output_shapes

:
df
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*
_output_shapes

:
db
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*
_output_shapes

:
dL

while/TanhTanhwhile/add_2:z:0*
T0*
_output_shapes

:
dc
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes

:
dP
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*
_output_shapes

:
dZ
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*
_output_shapes

:
d_
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*
_output_shapes

:
d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: V
while/Identity_4Identitywhile/add_3:z:0*
T0*
_output_shapes

:
d"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :
d: : :	d�:�:	d�:�:D
@

_output_shapes	
:�
!
_user_specified_name	unstack:Q	M

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:D@

_output_shapes	
:�
!
_user_specified_name	unstack:GC

_output_shapes
:	d�
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
��
�

<__inference___backward_gpu_gru_with_fallback_2375155_2375290
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat5
1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn=
9gradients_transpose_grad_invertpermutation_transpose_perm)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�U
gradients/grad_ys_0Identityplaceholder*
T0*
_output_shapes

:
dd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:
���������dW
gradients/grad_ys_2Identityplaceholder_2*
T0*
_output_shapes

:
dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::���
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:���������
d*
shrink_axis_mask�
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:�
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������
dq
gradients/Squeeze_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
   d   �
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*"
_output_shapes
:
d�
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:���������
da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*C
_output_shapes1
/:���������
d:
d: :��*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:
���������dp
gradients/ExpandDims_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*
_output_shapes

:
d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�Ng
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:d�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:d�
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
gradients/split_2_grad/concatConcatV2)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	d��
gradients/split_1_grad/concatConcatV2(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d�m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ,  �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	�r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:
���������dk

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*
_output_shapes

:
df

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	d�h

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	d�i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	�"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:
d:
���������d:
d: :���������
d:: ::���������
d:
d: :��:
d:: ::::::: : : *<
api_implements*(gru_999d41c9-6a05-4f02-bc6c-cc5bbb6e46d4*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2375289*
go_backwards( *

time_major( :IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:LH
"
_output_shapes
:
d
"
_user_specified_name
CudnnRNN:D@

_output_shapes

:��
 
_user_specified_nameconcat:H
D

_output_shapes
: 
*
_user_specified_nameCudnnRNN/input_c:N	J
"
_output_shapes
:
d
$
_user_specified_name
ExpandDims:VR
+
_output_shapes
:���������
d
#
_user_specified_name	transpose:B>

_output_shapes
:
"
_user_specified_name
CudnnRNN:@<

_output_shapes
: 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:UQ
+
_output_shapes
:���������
d
"
_user_specified_name
CudnnRNN:

_output_shapes
: :$ 

_output_shapes

:
d:1-
+
_output_shapes
:
���������d:$  

_output_shapes

:
d
�	
�
%__inference_gru_layer_call_fn_2375465

inputs
unknown:
d
	unknown_0:	d�
	unknown_1:	d�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:
���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_gru_layer_call_and_return_conditional_losses_2374872s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:
���������d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:
���������d: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2375461:'#
!
_user_specified_name	2375459:'#
!
_user_specified_name	2375457:'#
!
_user_specified_name	2375455:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�;
�
 __inference_standard_gru_2374658

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:�:�*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_mask\
MatMulMatMulstrided_slice_1:output:0kernel*
T0*
_output_shapes
:	
�`
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	
�Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitV
MatMul_1MatMulinit_hrecurrent_kernel*
T0*
_output_shapes
:	
�d
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	
�S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitW
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes

:
dD
SigmoidSigmoidadd:z:0*
T0*
_output_shapes

:
dY
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes

:
dH
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes

:
dT
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes

:
dP
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes

:
d@
TanhTanh	add_2:z:0*
T0*
_output_shapes

:
dJ
mul_1MulSigmoid:y:0init_h*
T0*
_output_shapes

:
dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes

:
dH
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes

:
dM
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes

:
dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :
d: : :	d�:�:	d�:�* 
_read_only_resource_inputs
 *
bodyR
while_body_2374569*
condR
while_cond_2374568*M
output_shapes<
:: : : : :
d: : :	d�:�:	d�:�*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������
d*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:
���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?W
IdentityIdentitystrided_slice_2:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:
���������dO

Identity_2Identitywhile:output:4*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_1c4119e2-5f05-4727-bd5c-882c50a531ba*
api_preferred_deviceCPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�-
�
while_body_2374569
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
d*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*
_output_shapes
:	
�s
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*
_output_shapes
:	
�W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitz
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*
_output_shapes
:	
�y
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*
_output_shapes
:	
�Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_spliti
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*
_output_shapes

:
dP
while/SigmoidSigmoidwhile/add:z:0*
T0*
_output_shapes

:
dk
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*
_output_shapes

:
dT
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*
_output_shapes

:
df
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*
_output_shapes

:
db
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*
_output_shapes

:
dL

while/TanhTanhwhile/add_2:z:0*
T0*
_output_shapes

:
dc
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes

:
dP
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*
_output_shapes

:
dZ
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*
_output_shapes

:
d_
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*
_output_shapes

:
d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: V
while/Identity_4Identitywhile/add_3:z:0*
T0*
_output_shapes

:
d"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :
d: : :	d�:�:	d�:�:D
@

_output_shapes	
:�
!
_user_specified_name	unstack:Q	M

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:D@

_output_shapes	
:�
!
_user_specified_name	unstack:GC

_output_shapes
:	d�
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�

+__inference_embedding_layer_call_fn_2375417

inputs
unknown:Ld
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:
���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_layer_call_and_return_conditional_losses_2374500s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:
���������d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:
���������: 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2375413:O K
'
_output_shapes
:
���������
 
_user_specified_nameinputs
�D
�
"__inference__wrapped_model_2373725
embedding_input?
-sequential_embedding_embedding_lookup_2373329:Ld=
+sequential_gru_read_readvariableop_resource:
d@
-sequential_gru_read_1_readvariableop_resource:	d�@
-sequential_gru_read_2_readvariableop_resource:	d�@
-sequential_gru_read_3_readvariableop_resource:	�D
2sequential_dense_tensordot_readvariableop_resource:dL>
0sequential_dense_biasadd_readvariableop_resource:L
identity��'sequential/dense/BiasAdd/ReadVariableOp�)sequential/dense/Tensordot/ReadVariableOp�%sequential/embedding/embedding_lookup�sequential/gru/AssignVariableOp�"sequential/gru/Read/ReadVariableOp�$sequential/gru/Read_1/ReadVariableOp�$sequential/gru/Read_2/ReadVariableOp�$sequential/gru/Read_3/ReadVariableOps
sequential/embedding/CastCastembedding_input*

DstT0*

SrcT0*'
_output_shapes
:
����������
%sequential/embedding/embedding_lookupResourceGather-sequential_embedding_embedding_lookup_2373329sequential/embedding/Cast:y:0*
Tindices0*@
_class6
42loc:@sequential/embedding/embedding_lookup/2373329*+
_output_shapes
:
���������d*
dtype0�
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:
���������d�
"sequential/gru/Read/ReadVariableOpReadVariableOp+sequential_gru_read_readvariableop_resource*
_output_shapes

:
d*
dtype0x
sequential/gru/IdentityIdentity*sequential/gru/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:
d�
$sequential/gru/Read_1/ReadVariableOpReadVariableOp-sequential_gru_read_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0}
sequential/gru/Identity_1Identity,sequential/gru/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	d��
$sequential/gru/Read_2/ReadVariableOpReadVariableOp-sequential_gru_read_2_readvariableop_resource*
_output_shapes
:	d�*
dtype0}
sequential/gru/Identity_2Identity,sequential/gru/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	d��
$sequential/gru/Read_3/ReadVariableOpReadVariableOp-sequential_gru_read_3_readvariableop_resource*
_output_shapes
:	�*
dtype0}
sequential/gru/Identity_3Identity,sequential/gru/Read_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
sequential/gru/PartitionedCallPartitionedCall7sequential/embedding/embedding_lookup/Identity:output:0 sequential/gru/Identity:output:0"sequential/gru/Identity_1:output:0"sequential/gru/Identity_2:output:0"sequential/gru/Identity_3:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:
d:
���������d:
d: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2373485�
sequential/gru/AssignVariableOpAssignVariableOp+sequential_gru_read_readvariableop_resource'sequential/gru/PartitionedCall:output:2#^sequential/gru/Read/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:dL*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
 sequential/dense/Tensordot/ShapeShape'sequential/gru/PartitionedCall:output:1*
T0*
_output_shapes
::��j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$sequential/dense/Tensordot/transpose	Transpose'sequential/gru/PartitionedCall:output:1*sequential/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:
���������d�
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Ll
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Lj
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:
���������L�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:L*
dtype0�
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:
���������Lt
IdentityIdentity!sequential/dense/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:
���������L�
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp&^sequential/embedding/embedding_lookup ^sequential/gru/AssignVariableOp#^sequential/gru/Read/ReadVariableOp%^sequential/gru/Read_1/ReadVariableOp%^sequential/gru/Read_2/ReadVariableOp%^sequential/gru/Read_3/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:
���������: : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2B
sequential/gru/AssignVariableOpsequential/gru/AssignVariableOp2H
"sequential/gru/Read/ReadVariableOp"sequential/gru/Read/ReadVariableOp2L
$sequential/gru/Read_1/ReadVariableOp$sequential/gru/Read_1/ReadVariableOp2L
$sequential/gru/Read_2/ReadVariableOp$sequential/gru/Read_2/ReadVariableOp2L
$sequential/gru/Read_3/ReadVariableOp$sequential/gru/Read_3/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:'#
!
_user_specified_name	2373329:X T
'
_output_shapes
:
���������
)
_user_specified_nameembedding_input
��
�

<__inference___backward_gpu_gru_with_fallback_2373957_2374092
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat5
1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn=
9gradients_transpose_grad_invertpermutation_transpose_perm)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�U
gradients/grad_ys_0Identityplaceholder*
T0*
_output_shapes

:
dd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:
���������dW
gradients/grad_ys_2Identityplaceholder_2*
T0*
_output_shapes

:
dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::���
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:���������
d*
shrink_axis_mask�
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:�
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������
dq
gradients/Squeeze_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
   d   �
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*"
_output_shapes
:
d�
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:���������
da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*C
_output_shapes1
/:���������
d:
d: :��*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:
���������dp
gradients/ExpandDims_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*
_output_shapes

:
d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�Ng
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:d�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:d�
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
gradients/split_2_grad/concatConcatV2)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	d��
gradients/split_1_grad/concatConcatV2(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d�m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ,  �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	�r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:
���������dk

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*
_output_shapes

:
df

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	d�h

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	d�i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	�"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:
d:
���������d:
d: :���������
d:: ::���������
d:
d: :��:
d:: ::::::: : : *<
api_implements*(gru_0660cc53-f4b6-450f-8c94-e84ee22a46c0*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2374091*
go_backwards( *

time_major( :IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:LH
"
_output_shapes
:
d
"
_user_specified_name
CudnnRNN:D@

_output_shapes

:��
 
_user_specified_nameconcat:H
D

_output_shapes
: 
*
_user_specified_nameCudnnRNN/input_c:N	J
"
_output_shapes
:
d
$
_user_specified_name
ExpandDims:VR
+
_output_shapes
:���������
d
#
_user_specified_name	transpose:B>

_output_shapes
:
"
_user_specified_name
CudnnRNN:@<

_output_shapes
: 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:UQ
+
_output_shapes
:���������
d
"
_user_specified_name
CudnnRNN:

_output_shapes
: :$ 

_output_shapes

:
d:1-
+
_output_shapes
:
���������d:$  

_output_shapes

:
d
�=
�
'__forward_gpu_gru_with_fallback_2374869

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1
	transpose

expanddims
cudnnrnn_input_c

concat

cudnnrnn_2
transpose_perm
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "#
concat_axisconcat/axis:output:0"
concatconcat_0:output:0"!

cudnnrnn_0CudnnRNN:output_c:0"&

cudnnrnn_1CudnnRNN:reserve_space:0"!

cudnnrnn_2CudnnRNN:output_h:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"
cudnnrnnCudnnRNN:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_1c4119e2-5f05-4727-bd5c-882c50a531ba*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2374735_2374870*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�
�
@__inference_gru_layer_call_and_return_conditional_losses_2375847
inputs_0.
read_readvariableop_resource:
d1
read_1_readvariableop_resource:	d�1
read_2_readvariableop_resource:	d�1
read_3_readvariableop_resource:	�

identity_4��AssignVariableOp�Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp�Read_3/ReadVariableOpp
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:
d*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:
du
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	d�*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�u
Read_3/ReadVariableOpReadVariableOpread_3_readvariableop_resource*
_output_shapes
:	�*
dtype0_

Identity_3IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
PartitionedCallPartitionedCallinputs_0Identity:output:0Identity_1:output:0Identity_2:output:0Identity_3:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:
d:
���������d:
d: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2375633�
AssignVariableOpAssignVariableOpread_readvariableop_resourcePartitionedCall:output:2^Read/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(m

Identity_4IdentityPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:
���������d�
NoOpNoOp^AssignVariableOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp^Read_3/ReadVariableOp*
_output_shapes
 "!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:
���������d: : : : 2$
AssignVariableOpAssignVariableOp2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp2.
Read_3/ReadVariableOpRead_3/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:U Q
+
_output_shapes
:
���������d
"
_user_specified_name
inputs_0
�-
�
while_body_2373396
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
d*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*
_output_shapes
:	
�s
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*
_output_shapes
:	
�W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitz
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*
_output_shapes
:	
�y
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*
_output_shapes
:	
�Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_spliti
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*
_output_shapes

:
dP
while/SigmoidSigmoidwhile/add:z:0*
T0*
_output_shapes

:
dk
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*
_output_shapes

:
dT
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*
_output_shapes

:
df
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*
_output_shapes

:
db
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*
_output_shapes

:
dL

while/TanhTanhwhile/add_2:z:0*
T0*
_output_shapes

:
dc
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes

:
dP
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*
_output_shapes

:
dZ
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*
_output_shapes

:
d_
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*
_output_shapes

:
d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: V
while/Identity_4Identitywhile/add_3:z:0*
T0*
_output_shapes

:
d"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :
d: : :	d�:�:	d�:�:D
@

_output_shapes	
:�
!
_user_specified_name	unstack:Q	M

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:D@

_output_shapes	
:�
!
_user_specified_name	unstack:GC

_output_shapes
:	d�
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
�
,__inference_sequential_layer_call_fn_2375346
embedding_input
unknown:Ld
	unknown_0:
d
	unknown_1:	d�
	unknown_2:	d�
	unknown_3:	�
	unknown_4:dL
	unknown_5:L
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:
���������L*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_2375308s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:
���������L<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:
���������: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2375342:'#
!
_user_specified_name	2375340:'#
!
_user_specified_name	2375338:'#
!
_user_specified_name	2375336:'#
!
_user_specified_name	2375334:'#
!
_user_specified_name	2375332:'#
!
_user_specified_name	2375330:X T
'
_output_shapes
:
���������
)
_user_specified_nameembedding_input
�

�
while_cond_2376281
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2376281___redundant_placeholder05
1while_while_cond_2376281___redundant_placeholder15
1while_while_cond_2376281___redundant_placeholder25
1while_while_cond_2376281___redundant_placeholder35
1while_while_cond_2376281___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: : : : :
d: ::::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�

�
while_cond_2375912
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2375912___redundant_placeholder05
1while_while_cond_2375912___redundant_placeholder15
1while_while_cond_2375912___redundant_placeholder25
1while_while_cond_2375912___redundant_placeholder35
1while_while_cond_2375912___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: : : : :
d: ::::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�4
�
)__inference_gpu_gru_with_fallback_2374325

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:��U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_1d4607a8-8de1-46eb-965d-c935dbf59148*
api_preferred_deviceGPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�=
�
'__forward_gpu_gru_with_fallback_2374460

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1
	transpose

expanddims
cudnnrnn_input_c

concat

cudnnrnn_2
transpose_perm
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "#
concat_axisconcat/axis:output:0"
concatconcat_0:output:0"!

cudnnrnn_0CudnnRNN:output_c:0"&

cudnnrnn_1CudnnRNN:reserve_space:0"!

cudnnrnn_2CudnnRNN:output_h:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"
cudnnrnnCudnnRNN:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_1d4607a8-8de1-46eb-965d-c935dbf59148*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2374326_2374461*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�
�
G__inference_sequential_layer_call_and_return_conditional_losses_2375308
embedding_input#
embedding_2374921:Ld
gru_2375293:
d
gru_2375295:	d�
gru_2375297:	d�
gru_2375299:	�
dense_2375302:dL
dense_2375304:L
identity��dense/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�gru/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_2374921*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:
���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_layer_call_and_return_conditional_losses_2374500�
gru/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0gru_2375293gru_2375295gru_2375297gru_2375299*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:
���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_gru_layer_call_and_return_conditional_losses_2375292�
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_2375302dense_2375304*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:
���������L*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_2374911y
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:
���������L�
NoOpNoOp^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^gru/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:
���������: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:'#
!
_user_specified_name	2375304:'#
!
_user_specified_name	2375302:'#
!
_user_specified_name	2375299:'#
!
_user_specified_name	2375297:'#
!
_user_specified_name	2375295:'#
!
_user_specified_name	2375293:'#
!
_user_specified_name	2374921:X T
'
_output_shapes
:
���������
)
_user_specified_nameembedding_input
�=
�
'__forward_gpu_gru_with_fallback_2376951

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1
	transpose

expanddims
cudnnrnn_input_c

concat

cudnnrnn_2
transpose_perm
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "#
concat_axisconcat/axis:output:0"
concatconcat_0:output:0"!

cudnnrnn_0CudnnRNN:output_c:0"&

cudnnrnn_1CudnnRNN:reserve_space:0"!

cudnnrnn_2CudnnRNN:output_h:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"
cudnnrnnCudnnRNN:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_7393682a-ed09-4efb-89b0-ce0fc0b05b91*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2376817_2376952*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�
�
B__inference_dense_layer_call_and_return_conditional_losses_2376993

inputs3
!tensordot_readvariableop_resource:dL-
biasadd_readvariableop_resource:L
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:dL*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:
���������d�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������L[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:LY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:
���������Lr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:L*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:
���������Lc
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:
���������LV
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:
���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�
�
@__inference_gru_layer_call_and_return_conditional_losses_2376216
inputs_0.
read_readvariableop_resource:
d1
read_1_readvariableop_resource:	d�1
read_2_readvariableop_resource:	d�1
read_3_readvariableop_resource:	�

identity_4��AssignVariableOp�Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp�Read_3/ReadVariableOpp
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:
d*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:
du
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	d�*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�u
Read_3/ReadVariableOpReadVariableOpread_3_readvariableop_resource*
_output_shapes
:	�*
dtype0_

Identity_3IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
PartitionedCallPartitionedCallinputs_0Identity:output:0Identity_1:output:0Identity_2:output:0Identity_3:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:
d:
���������d:
d: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2376002�
AssignVariableOpAssignVariableOpread_readvariableop_resourcePartitionedCall:output:2^Read/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(m

Identity_4IdentityPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:
���������d�
NoOpNoOp^AssignVariableOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp^Read_3/ReadVariableOp*
_output_shapes
 "!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:
���������d: : : : 2$
AssignVariableOpAssignVariableOp2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp2.
Read_3/ReadVariableOpRead_3/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:U Q
+
_output_shapes
:
���������d
"
_user_specified_name
inputs_0
�-
�
while_body_2373791
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
d*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*
_output_shapes
:	
�s
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*
_output_shapes
:	
�W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitz
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*
_output_shapes
:	
�y
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*
_output_shapes
:	
�Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_spliti
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*
_output_shapes

:
dP
while/SigmoidSigmoidwhile/add:z:0*
T0*
_output_shapes

:
dk
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*
_output_shapes

:
dT
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*
_output_shapes

:
df
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*
_output_shapes

:
db
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*
_output_shapes

:
dL

while/TanhTanhwhile/add_2:z:0*
T0*
_output_shapes

:
dc
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes

:
dP
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*
_output_shapes

:
dZ
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*
_output_shapes

:
d_
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*
_output_shapes

:
d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: V
while/Identity_4Identitywhile/add_3:z:0*
T0*
_output_shapes

:
d"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :
d: : :	d�:�:	d�:�:D
@

_output_shapes	
:�
!
_user_specified_name	unstack:Q	M

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:D@

_output_shapes	
:�
!
_user_specified_name	unstack:GC

_output_shapes
:	d�
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
�
G__inference_sequential_layer_call_and_return_conditional_losses_2374918
embedding_input#
embedding_2374501:Ld
gru_2374873:
d
gru_2374875:	d�
gru_2374877:	d�
gru_2374879:	�
dense_2374912:dL
dense_2374914:L
identity��dense/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�gru/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_2374501*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:
���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_layer_call_and_return_conditional_losses_2374500�
gru/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0gru_2374873gru_2374875gru_2374877gru_2374879*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:
���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_gru_layer_call_and_return_conditional_losses_2374872�
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_2374912dense_2374914*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:
���������L*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_2374911y
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:
���������L�
NoOpNoOp^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^gru/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:
���������: : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:'#
!
_user_specified_name	2374914:'#
!
_user_specified_name	2374912:'#
!
_user_specified_name	2374879:'#
!
_user_specified_name	2374877:'#
!
_user_specified_name	2374875:'#
!
_user_specified_name	2374873:'#
!
_user_specified_name	2374501:X T
'
_output_shapes
:
���������
)
_user_specified_nameembedding_input
�

�
while_cond_2376650
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2376650___redundant_placeholder05
1while_while_cond_2376650___redundant_placeholder15
1while_while_cond_2376650___redundant_placeholder25
1while_while_cond_2376650___redundant_placeholder35
1while_while_cond_2376650___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: : : : :
d: ::::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�=
�
'__forward_gpu_gru_with_fallback_2375289

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1
	transpose

expanddims
cudnnrnn_input_c

concat

cudnnrnn_2
transpose_perm
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "#
concat_axisconcat/axis:output:0"
concatconcat_0:output:0"!

cudnnrnn_0CudnnRNN:output_c:0"&

cudnnrnn_1CudnnRNN:reserve_space:0"!

cudnnrnn_2CudnnRNN:output_h:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"
cudnnrnnCudnnRNN:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_999d41c9-6a05-4f02-bc6c-cc5bbb6e46d4*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2375155_2375290*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�=
�
'__forward_gpu_gru_with_fallback_2373696

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1
	transpose

expanddims
cudnnrnn_input_c

concat

cudnnrnn_2
transpose_perm
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "#
concat_axisconcat/axis:output:0"
concatconcat_0:output:0"!

cudnnrnn_0CudnnRNN:output_c:0"&

cudnnrnn_1CudnnRNN:reserve_space:0"!

cudnnrnn_2CudnnRNN:output_h:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"
cudnnrnnCudnnRNN:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_fd8c5e29-b8f5-421c-87aa-838be1d67593*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2373562_2373697*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�4
�
)__inference_gpu_gru_with_fallback_2373561

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:��U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_fd8c5e29-b8f5-421c-87aa-838be1d67593*
api_preferred_deviceGPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�;
�
 __inference_standard_gru_2376740

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:�:�*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_mask\
MatMulMatMulstrided_slice_1:output:0kernel*
T0*
_output_shapes
:	
�`
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	
�Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitV
MatMul_1MatMulinit_hrecurrent_kernel*
T0*
_output_shapes
:	
�d
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	
�S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitW
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes

:
dD
SigmoidSigmoidadd:z:0*
T0*
_output_shapes

:
dY
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes

:
dH
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes

:
dT
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes

:
dP
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes

:
d@
TanhTanh	add_2:z:0*
T0*
_output_shapes

:
dJ
mul_1MulSigmoid:y:0init_h*
T0*
_output_shapes

:
dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes

:
dH
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes

:
dM
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes

:
dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :
d: : :	d�:�:	d�:�* 
_read_only_resource_inputs
 *
bodyR
while_body_2376651*
condR
while_cond_2376650*M
output_shapes<
:: : : : :
d: : :	d�:�:	d�:�*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������
d*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:
���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?W
IdentityIdentitystrided_slice_2:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:
���������dO

Identity_2Identitywhile:output:4*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_7393682a-ed09-4efb-89b0-ce0fc0b05b91*
api_preferred_deviceCPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�;
�
 __inference_standard_gru_2376371

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:�:�*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_mask\
MatMulMatMulstrided_slice_1:output:0kernel*
T0*
_output_shapes
:	
�`
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	
�Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitV
MatMul_1MatMulinit_hrecurrent_kernel*
T0*
_output_shapes
:	
�d
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	
�S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitW
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes

:
dD
SigmoidSigmoidadd:z:0*
T0*
_output_shapes

:
dY
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes

:
dH
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes

:
dT
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes

:
dP
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes

:
d@
TanhTanh	add_2:z:0*
T0*
_output_shapes

:
dJ
mul_1MulSigmoid:y:0init_h*
T0*
_output_shapes

:
dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes

:
dH
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes

:
dM
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes

:
dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :
d: : :	d�:�:	d�:�* 
_read_only_resource_inputs
 *
bodyR
while_body_2376282*
condR
while_cond_2376281*M
output_shapes<
:: : : : :
d: : :	d�:�:	d�:�*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������
d*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:
���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?W
IdentityIdentitystrided_slice_2:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:
���������dO

Identity_2Identitywhile:output:4*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_d0df98bb-a0f8-4a1d-a383-06362d752882*
api_preferred_deviceCPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�	
�
%__inference_gru_layer_call_fn_2375478

inputs
unknown:
d
	unknown_0:	d�
	unknown_1:	d�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:
���������d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_gru_layer_call_and_return_conditional_losses_2375292s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:
���������d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:
���������d: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2375474:'#
!
_user_specified_name	2375472:'#
!
_user_specified_name	2375470:'#
!
_user_specified_name	2375468:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�;
�
 __inference_standard_gru_2375633

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:�:�*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_mask\
MatMulMatMulstrided_slice_1:output:0kernel*
T0*
_output_shapes
:	
�`
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	
�Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitV
MatMul_1MatMulinit_hrecurrent_kernel*
T0*
_output_shapes
:	
�d
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	
�S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitW
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes

:
dD
SigmoidSigmoidadd:z:0*
T0*
_output_shapes

:
dY
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes

:
dH
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes

:
dT
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes

:
dP
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes

:
d@
TanhTanh	add_2:z:0*
T0*
_output_shapes

:
dJ
mul_1MulSigmoid:y:0init_h*
T0*
_output_shapes

:
dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes

:
dH
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes

:
dM
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes

:
dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :
d: : :	d�:�:	d�:�* 
_read_only_resource_inputs
 *
bodyR
while_body_2375544*
condR
while_cond_2375543*M
output_shapes<
:: : : : :
d: : :	d�:�:	d�:�*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������
d*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:
���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?W
IdentityIdentitystrided_slice_2:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:
���������dO

Identity_2Identitywhile:output:4*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_3378f1f8-d8c8-407b-8966-68b8acf9ba31*
api_preferred_deviceCPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
��
�

<__inference___backward_gpu_gru_with_fallback_2374735_2374870
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat5
1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn=
9gradients_transpose_grad_invertpermutation_transpose_perm)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�U
gradients/grad_ys_0Identityplaceholder*
T0*
_output_shapes

:
dd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:
���������dW
gradients/grad_ys_2Identityplaceholder_2*
T0*
_output_shapes

:
dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::���
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:���������
d*
shrink_axis_mask�
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:�
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������
dq
gradients/Squeeze_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
   d   �
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*"
_output_shapes
:
d�
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:���������
da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*C
_output_shapes1
/:���������
d:
d: :��*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:
���������dp
gradients/ExpandDims_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*
_output_shapes

:
d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�Ng
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:d�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:d�
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
gradients/split_2_grad/concatConcatV2)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	d��
gradients/split_1_grad/concatConcatV2(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d�m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ,  �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	�r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:
���������dk

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*
_output_shapes

:
df

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	d�h

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	d�i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	�"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:
d:
���������d:
d: :���������
d:: ::���������
d:
d: :��:
d:: ::::::: : : *<
api_implements*(gru_1c4119e2-5f05-4727-bd5c-882c50a531ba*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2374869*
go_backwards( *

time_major( :IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:LH
"
_output_shapes
:
d
"
_user_specified_name
CudnnRNN:D@

_output_shapes

:��
 
_user_specified_nameconcat:H
D

_output_shapes
: 
*
_user_specified_nameCudnnRNN/input_c:N	J
"
_output_shapes
:
d
$
_user_specified_name
ExpandDims:VR
+
_output_shapes
:���������
d
#
_user_specified_name	transpose:B>

_output_shapes
:
"
_user_specified_name
CudnnRNN:@<

_output_shapes
: 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:UQ
+
_output_shapes
:���������
d
"
_user_specified_name
CudnnRNN:

_output_shapes
: :$ 

_output_shapes

:
d:1-
+
_output_shapes
:
���������d:$  

_output_shapes

:
d
�
�
@__inference_gru_layer_call_and_return_conditional_losses_2374463

inputs.
read_readvariableop_resource:
d1
read_1_readvariableop_resource:	d�1
read_2_readvariableop_resource:	d�1
read_3_readvariableop_resource:	�

identity_4��AssignVariableOp�Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp�Read_3/ReadVariableOpp
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:
d*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:
du
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	d�*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�u
Read_3/ReadVariableOpReadVariableOpread_3_readvariableop_resource*
_output_shapes
:	�*
dtype0_

Identity_3IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
PartitionedCallPartitionedCallinputsIdentity:output:0Identity_1:output:0Identity_2:output:0Identity_3:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:
d:
���������d:
d: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2374249�
AssignVariableOpAssignVariableOpread_readvariableop_resourcePartitionedCall:output:2^Read/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(m

Identity_4IdentityPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:
���������d�
NoOpNoOp^AssignVariableOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp^Read_3/ReadVariableOp*
_output_shapes
 "!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:
���������d: : : : 2$
AssignVariableOpAssignVariableOp2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp2.
Read_3/ReadVariableOpRead_3/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�=
�
'__forward_gpu_gru_with_fallback_2374091

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1
	transpose

expanddims
cudnnrnn_input_c

concat

cudnnrnn_2
transpose_perm
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "#
concat_axisconcat/axis:output:0"
concatconcat_0:output:0"!

cudnnrnn_0CudnnRNN:output_c:0"&

cudnnrnn_1CudnnRNN:reserve_space:0"!

cudnnrnn_2CudnnRNN:output_h:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"
cudnnrnnCudnnRNN:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_0660cc53-f4b6-450f-8c94-e84ee22a46c0*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2373957_2374092*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�
�
 __inference__traced_save_2377153
file_prefix=
+read_disablecopyonread_embedding_embeddings:Ld7
%read_1_disablecopyonread_dense_kernel:dL1
#read_2_disablecopyonread_dense_bias:L?
,read_3_disablecopyonread_gru_gru_cell_kernel:	d�I
6read_4_disablecopyonread_gru_gru_cell_recurrent_kernel:	d�=
*read_5_disablecopyonread_gru_gru_cell_bias:	�,
"read_6_disablecopyonread_iteration:	 0
&read_7_disablecopyonread_learning_rate: F
4read_8_disablecopyonread_adam_m_embedding_embeddings:LdF
4read_9_disablecopyonread_adam_v_embedding_embeddings:LdG
4read_10_disablecopyonread_adam_m_gru_gru_cell_kernel:	d�G
4read_11_disablecopyonread_adam_v_gru_gru_cell_kernel:	d�Q
>read_12_disablecopyonread_adam_m_gru_gru_cell_recurrent_kernel:	d�Q
>read_13_disablecopyonread_adam_v_gru_gru_cell_recurrent_kernel:	d�E
2read_14_disablecopyonread_adam_m_gru_gru_cell_bias:	�E
2read_15_disablecopyonread_adam_v_gru_gru_cell_bias:	�?
-read_16_disablecopyonread_adam_m_dense_kernel:dL?
-read_17_disablecopyonread_adam_v_dense_kernel:dL9
+read_18_disablecopyonread_adam_m_dense_bias:L9
+read_19_disablecopyonread_adam_v_dense_bias:L8
&read_20_disablecopyonread_gru_variable:
d)
read_21_disablecopyonread_total: )
read_22_disablecopyonread_count: 
savev2_const
identity_47��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: }
Read/DisableCopyOnReadDisableCopyOnRead+read_disablecopyonread_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp+read_disablecopyonread_embedding_embeddings^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:Ld*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:Lda

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:Ldy
Read_1/DisableCopyOnReadDisableCopyOnRead%read_1_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp%read_1_disablecopyonread_dense_kernel^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:dL*
dtype0m

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:dLc

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes

:dLw
Read_2/DisableCopyOnReadDisableCopyOnRead#read_2_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp#read_2_disablecopyonread_dense_bias^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:L*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:L_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:L�
Read_3/DisableCopyOnReadDisableCopyOnRead,read_3_disablecopyonread_gru_gru_cell_kernel"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp,read_3_disablecopyonread_gru_gru_cell_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	d�*
dtype0n

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	d�d

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:	d��
Read_4/DisableCopyOnReadDisableCopyOnRead6read_4_disablecopyonread_gru_gru_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp6read_4_disablecopyonread_gru_gru_cell_recurrent_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	d�*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	d�d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	d�~
Read_5/DisableCopyOnReadDisableCopyOnRead*read_5_disablecopyonread_gru_gru_cell_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp*read_5_disablecopyonread_gru_gru_cell_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:	�v
Read_6/DisableCopyOnReadDisableCopyOnRead"read_6_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp"read_6_disablecopyonread_iteration^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_learning_rate^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_8/DisableCopyOnReadDisableCopyOnRead4read_8_disablecopyonread_adam_m_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp4read_8_disablecopyonread_adam_m_embedding_embeddings^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:Ld*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:Lde
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:Ld�
Read_9/DisableCopyOnReadDisableCopyOnRead4read_9_disablecopyonread_adam_v_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp4read_9_disablecopyonread_adam_v_embedding_embeddings^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:Ld*
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:Lde
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

:Ld�
Read_10/DisableCopyOnReadDisableCopyOnRead4read_10_disablecopyonread_adam_m_gru_gru_cell_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp4read_10_disablecopyonread_adam_m_gru_gru_cell_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	d�*
dtype0p
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	d�f
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	d��
Read_11/DisableCopyOnReadDisableCopyOnRead4read_11_disablecopyonread_adam_v_gru_gru_cell_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp4read_11_disablecopyonread_adam_v_gru_gru_cell_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	d�*
dtype0p
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	d�f
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:	d��
Read_12/DisableCopyOnReadDisableCopyOnRead>read_12_disablecopyonread_adam_m_gru_gru_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp>read_12_disablecopyonread_adam_m_gru_gru_cell_recurrent_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	d�*
dtype0p
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	d�f
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	d��
Read_13/DisableCopyOnReadDisableCopyOnRead>read_13_disablecopyonread_adam_v_gru_gru_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp>read_13_disablecopyonread_adam_v_gru_gru_cell_recurrent_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	d�*
dtype0p
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	d�f
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:	d��
Read_14/DisableCopyOnReadDisableCopyOnRead2read_14_disablecopyonread_adam_m_gru_gru_cell_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp2read_14_disablecopyonread_adam_m_gru_gru_cell_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_15/DisableCopyOnReadDisableCopyOnRead2read_15_disablecopyonread_adam_v_gru_gru_cell_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp2read_15_disablecopyonread_adam_v_gru_gru_cell_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_16/DisableCopyOnReadDisableCopyOnRead-read_16_disablecopyonread_adam_m_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp-read_16_disablecopyonread_adam_m_dense_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:dL*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:dLe
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:dL�
Read_17/DisableCopyOnReadDisableCopyOnRead-read_17_disablecopyonread_adam_v_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp-read_17_disablecopyonread_adam_v_dense_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:dL*
dtype0o
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:dLe
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes

:dL�
Read_18/DisableCopyOnReadDisableCopyOnRead+read_18_disablecopyonread_adam_m_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp+read_18_disablecopyonread_adam_m_dense_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:L*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:La
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:L�
Read_19/DisableCopyOnReadDisableCopyOnRead+read_19_disablecopyonread_adam_v_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp+read_19_disablecopyonread_adam_v_dense_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:L*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:La
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:L{
Read_20/DisableCopyOnReadDisableCopyOnRead&read_20_disablecopyonread_gru_variable"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp&read_20_disablecopyonread_gru_variable^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
d*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
de
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:
dt
Read_21/DisableCopyOnReadDisableCopyOnReadread_21_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOpread_21_disablecopyonread_total^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_22/DisableCopyOnReadDisableCopyOnReadread_22_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOpread_22_disablecopyonread_count^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: �

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-1/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *&
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_46Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_47IdentityIdentity_46:output:0^NoOp*
T0*
_output_shapes
: �	
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_47Identity_47:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:%!

_user_specified_namecount:%!

_user_specified_nametotal:,(
&
_user_specified_namegru/Variable:1-
+
_user_specified_nameAdam/v/dense/bias:1-
+
_user_specified_nameAdam/m/dense/bias:3/
-
_user_specified_nameAdam/v/dense/kernel:3/
-
_user_specified_nameAdam/m/dense/kernel:84
2
_user_specified_nameAdam/v/gru/gru_cell/bias:84
2
_user_specified_nameAdam/m/gru/gru_cell/bias:D@
>
_user_specified_name&$Adam/v/gru/gru_cell/recurrent_kernel:D@
>
_user_specified_name&$Adam/m/gru/gru_cell/recurrent_kernel::6
4
_user_specified_nameAdam/v/gru/gru_cell/kernel::6
4
_user_specified_nameAdam/m/gru/gru_cell/kernel:;
7
5
_user_specified_nameAdam/v/embedding/embeddings:;	7
5
_user_specified_nameAdam/m/embedding/embeddings:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:1-
+
_user_specified_namegru/gru_cell/bias:=9
7
_user_specified_namegru/gru_cell/recurrent_kernel:3/
-
_user_specified_namegru/gru_cell/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:40
.
_user_specified_nameembedding/embeddings:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
,__inference_sequential_layer_call_fn_2375327
embedding_input
unknown:Ld
	unknown_0:
d
	unknown_1:	d�
	unknown_2:	d�
	unknown_3:	�
	unknown_4:dL
	unknown_5:L
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:
���������L*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_2374918s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:
���������L<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:
���������: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2375323:'#
!
_user_specified_name	2375321:'#
!
_user_specified_name	2375319:'#
!
_user_specified_name	2375317:'#
!
_user_specified_name	2375315:'#
!
_user_specified_name	2375313:'#
!
_user_specified_name	2375311:X T
'
_output_shapes
:
���������
)
_user_specified_nameembedding_input
�4
�
)__inference_gpu_gru_with_fallback_2374734

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:��U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_1c4119e2-5f05-4727-bd5c-882c50a531ba*
api_preferred_deviceGPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�=
�
'__forward_gpu_gru_with_fallback_2376582

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1
	transpose

expanddims
cudnnrnn_input_c

concat

cudnnrnn_2
transpose_perm
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "#
concat_axisconcat/axis:output:0"
concatconcat_0:output:0"!

cudnnrnn_0CudnnRNN:output_c:0"&

cudnnrnn_1CudnnRNN:reserve_space:0"!

cudnnrnn_2CudnnRNN:output_h:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"
cudnnrnnCudnnRNN:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_d0df98bb-a0f8-4a1d-a383-06362d752882*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2376448_2376583*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�4
�
)__inference_gpu_gru_with_fallback_2373956

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:��U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_0660cc53-f4b6-450f-8c94-e84ee22a46c0*
api_preferred_deviceGPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�4
�
)__inference_gpu_gru_with_fallback_2375154

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:��U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_999d41c9-6a05-4f02-bc6c-cc5bbb6e46d4*
api_preferred_deviceGPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�
�
F__inference_embedding_layer_call_and_return_conditional_losses_2374500

inputs*
embedding_lookup_2374495:Ld
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:
����������
embedding_lookupResourceGatherembedding_lookup_2374495Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/2374495*+
_output_shapes
:
���������d*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:
���������du
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:
���������d5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:
���������: 2$
embedding_lookupembedding_lookup:'#
!
_user_specified_name	2374495:O K
'
_output_shapes
:
���������
 
_user_specified_nameinputs
�
�
@__inference_gru_layer_call_and_return_conditional_losses_2376954

inputs.
read_readvariableop_resource:
d1
read_1_readvariableop_resource:	d�1
read_2_readvariableop_resource:	d�1
read_3_readvariableop_resource:	�

identity_4��AssignVariableOp�Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp�Read_3/ReadVariableOpp
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:
d*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:
du
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	d�*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�u
Read_3/ReadVariableOpReadVariableOpread_3_readvariableop_resource*
_output_shapes
:	�*
dtype0_

Identity_3IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
PartitionedCallPartitionedCallinputsIdentity:output:0Identity_1:output:0Identity_2:output:0Identity_3:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:
d:
���������d:
d: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2376740�
AssignVariableOpAssignVariableOpread_readvariableop_resourcePartitionedCall:output:2^Read/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(m

Identity_4IdentityPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:
���������d�
NoOpNoOp^AssignVariableOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp^Read_3/ReadVariableOp*
_output_shapes
 "!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:
���������d: : : : 2$
AssignVariableOpAssignVariableOp2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp2.
Read_3/ReadVariableOpRead_3/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�-
�
while_body_2376282
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
d*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*
_output_shapes
:	
�s
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*
_output_shapes
:	
�W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitz
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*
_output_shapes
:	
�y
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*
_output_shapes
:	
�Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_spliti
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*
_output_shapes

:
dP
while/SigmoidSigmoidwhile/add:z:0*
T0*
_output_shapes

:
dk
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*
_output_shapes

:
dT
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*
_output_shapes

:
df
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*
_output_shapes

:
db
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*
_output_shapes

:
dL

while/TanhTanhwhile/add_2:z:0*
T0*
_output_shapes

:
dc
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes

:
dP
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*
_output_shapes

:
dZ
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*
_output_shapes

:
d_
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*
_output_shapes

:
d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: V
while/Identity_4Identitywhile/add_3:z:0*
T0*
_output_shapes

:
d"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :
d: : :	d�:�:	d�:�:D
@

_output_shapes	
:�
!
_user_specified_name	unstack:Q	M

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:D@

_output_shapes	
:�
!
_user_specified_name	unstack:GC

_output_shapes
:	d�
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�;
�
 __inference_standard_gru_2373485

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3O
unstackUnpackbias*
T0*"
_output_shapes
:�:�*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_mask\
MatMulMatMulstrided_slice_1:output:0kernel*
T0*
_output_shapes
:	
�`
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	
�Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitV
MatMul_1MatMulinit_hrecurrent_kernel*
T0*
_output_shapes
:	
�d
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	
�S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitW
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes

:
dD
SigmoidSigmoidadd:z:0*
T0*
_output_shapes

:
dY
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes

:
dH
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes

:
dT
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes

:
dP
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes

:
d@
TanhTanh	add_2:z:0*
T0*
_output_shapes

:
dJ
mul_1MulSigmoid:y:0init_h*
T0*
_output_shapes

:
dJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?P
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes

:
dH
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes

:
dM
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes

:
dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :
d: : :	d�:�:	d�:�* 
_read_only_resource_inputs
 *
bodyR
while_body_2373396*
condR
while_cond_2373395*M
output_shapes<
:: : : : :
d: : :	d�:�:	d�:�*
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������
d*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:
���������d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?W
IdentityIdentitystrided_slice_2:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:
���������dO

Identity_2Identitywhile:output:4*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_fd8c5e29-b8f5-421c-87aa-838be1d67593*
api_preferred_deviceCPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�-
�
while_body_2375544
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
d*
element_dtype0�
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*
_output_shapes
:	
�s
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*
_output_shapes
:	
�W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_splitz
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*
_output_shapes
:	
�y
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*
_output_shapes
:	
�Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*2
_output_shapes 
:
d:
d:
d*
	num_spliti
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*
_output_shapes

:
dP
while/SigmoidSigmoidwhile/add:z:0*
T0*
_output_shapes

:
dk
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*
_output_shapes

:
dT
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*
_output_shapes

:
df
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*
_output_shapes

:
db
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*
_output_shapes

:
dL

while/TanhTanhwhile/add_2:z:0*
T0*
_output_shapes

:
dc
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes

:
dP
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*
_output_shapes

:
dZ
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*
_output_shapes

:
d_
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*
_output_shapes

:
d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype0:���O
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: V
while/Identity_4Identitywhile/add_3:z:0*
T0*
_output_shapes

:
d"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :
d: : :	d�:�:	d�:�:D
@

_output_shapes	
:�
!
_user_specified_name	unstack:Q	M

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:D@

_output_shapes	
:�
!
_user_specified_name	unstack:GC

_output_shapes
:	d�
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
��
�

<__inference___backward_gpu_gru_with_fallback_2373562_2373697
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat5
1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn=
9gradients_transpose_grad_invertpermutation_transpose_perm)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�U
gradients/grad_ys_0Identityplaceholder*
T0*
_output_shapes

:
dd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:
���������dW
gradients/grad_ys_2Identityplaceholder_2*
T0*
_output_shapes

:
dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::���
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:���������
d*
shrink_axis_mask�
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:�
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������
dq
gradients/Squeeze_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
   d   �
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*"
_output_shapes
:
d�
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:���������
da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*C
_output_shapes1
/:���������
d:
d: :��*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:
���������dp
gradients/ExpandDims_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*
_output_shapes

:
d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�Ng
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:d�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:d�
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
gradients/split_2_grad/concatConcatV2)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	d��
gradients/split_1_grad/concatConcatV2(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d�m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ,  �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	�r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:
���������dk

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*
_output_shapes

:
df

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	d�h

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	d�i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	�"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:
d:
���������d:
d: :���������
d:: ::���������
d:
d: :��:
d:: ::::::: : : *<
api_implements*(gru_fd8c5e29-b8f5-421c-87aa-838be1d67593*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2373696*
go_backwards( *

time_major( :IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:LH
"
_output_shapes
:
d
"
_user_specified_name
CudnnRNN:D@

_output_shapes

:��
 
_user_specified_nameconcat:H
D

_output_shapes
: 
*
_user_specified_nameCudnnRNN/input_c:N	J
"
_output_shapes
:
d
$
_user_specified_name
ExpandDims:VR
+
_output_shapes
:���������
d
#
_user_specified_name	transpose:B>

_output_shapes
:
"
_user_specified_name
CudnnRNN:@<

_output_shapes
: 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:UQ
+
_output_shapes
:���������
d
"
_user_specified_name
CudnnRNN:

_output_shapes
: :$ 

_output_shapes

:
d:1-
+
_output_shapes
:
���������d:$  

_output_shapes

:
d
�

�
while_cond_2375543
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_2375543___redundant_placeholder05
1while_while_cond_2375543___redundant_placeholder15
1while_while_cond_2375543___redundant_placeholder25
1while_while_cond_2375543___redundant_placeholder35
1while_while_cond_2375543___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: : : : :
d: ::::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:$ 

_output_shapes

:
d:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
��
�

<__inference___backward_gpu_gru_with_fallback_2376079_2376214
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat5
1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn=
9gradients_transpose_grad_invertpermutation_transpose_perm)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4�U
gradients/grad_ys_0Identityplaceholder*
T0*
_output_shapes

:
dd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:
���������dW
gradients/grad_ys_2Identityplaceholder_2*
T0*
_output_shapes

:
dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: �
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::���
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:�
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:���������
d*
shrink_axis_mask�
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:�
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:���������
dq
gradients/Squeeze_grad/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
   d   �
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*"
_output_shapes
:
d�
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:���������
da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:�
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn1gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*C
_output_shapes1
/:���������
d:
d: :��*
rnn_modegru�
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:�
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:
���������dp
gradients/ExpandDims_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"
   d   �
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*
_output_shapes

:
d\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:�Nh
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:�Ng
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:dg
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dh
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:d�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::�
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:�N�
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:d�
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:d�
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:d�
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:�
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:�
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:�
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:�
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:�
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:�
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:dd�
gradients/split_2_grad/concatConcatV2)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:��
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	d��
gradients/split_1_grad/concatConcatV2(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	d�m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ,  �
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	�r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:
���������dk

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*
_output_shapes

:
df

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	d�h

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	d�i

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	�"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:
d:
���������d:
d: :���������
d:: ::���������
d:
d: :��:
d:: ::::::: : : *<
api_implements*(gru_147464ad-d9c1-4bc4-af9c-aa73203e3343*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_2376213*
go_backwards( *

time_major( :IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:LH
"
_output_shapes
:
d
"
_user_specified_name
CudnnRNN:D@

_output_shapes

:��
 
_user_specified_nameconcat:H
D

_output_shapes
: 
*
_user_specified_nameCudnnRNN/input_c:N	J
"
_output_shapes
:
d
$
_user_specified_name
ExpandDims:VR
+
_output_shapes
:���������
d
#
_user_specified_name	transpose:B>

_output_shapes
:
"
_user_specified_name
CudnnRNN:@<

_output_shapes
: 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:UQ
+
_output_shapes
:���������
d
"
_user_specified_name
CudnnRNN:

_output_shapes
: :$ 

_output_shapes

:
d:1-
+
_output_shapes
:
���������d:$  

_output_shapes

:
d
�
�
@__inference_gru_layer_call_and_return_conditional_losses_2375292

inputs.
read_readvariableop_resource:
d1
read_1_readvariableop_resource:	d�1
read_2_readvariableop_resource:	d�1
read_3_readvariableop_resource:	�

identity_4��AssignVariableOp�Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp�Read_3/ReadVariableOpp
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:
d*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:
du
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	d�*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�u
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	d�*
dtype0_

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�u
Read_3/ReadVariableOpReadVariableOpread_3_readvariableop_resource*
_output_shapes
:	�*
dtype0_

Identity_3IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
PartitionedCallPartitionedCallinputsIdentity:output:0Identity_1:output:0Identity_2:output:0Identity_3:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:
d:
���������d:
d: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_standard_gru_2375078�
AssignVariableOpAssignVariableOpread_readvariableop_resourcePartitionedCall:output:2^Read/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(m

Identity_4IdentityPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:
���������d�
NoOpNoOp^AssignVariableOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp^Read_3/ReadVariableOp*
_output_shapes
 "!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:
���������d: : : : 2$
AssignVariableOpAssignVariableOp2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp2.
Read_3/ReadVariableOpRead_3/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�
�
F__inference_embedding_layer_call_and_return_conditional_losses_2375426

inputs*
embedding_lookup_2375421:Ld
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:
����������
embedding_lookupResourceGatherembedding_lookup_2375421Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/2375421*+
_output_shapes
:
���������d*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:
���������du
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:
���������d5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:
���������: 2$
embedding_lookupembedding_lookup:'#
!
_user_specified_name	2375421:O K
'
_output_shapes
:
���������
 
_user_specified_nameinputs
�4
�
)__inference_gpu_gru_with_fallback_2376816

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
dP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:��U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_7393682a-ed09-4efb-89b0-ce0fc0b05b91*
api_preferred_deviceGPU*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs
�=
�
'__forward_gpu_gru_with_fallback_2376213

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1
	transpose

expanddims
cudnnrnn_input_c

concat

cudnnrnn_2
transpose_perm
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim�c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : f

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*"
_output_shapes
:
dQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:dd:dd:dd*
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������V
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:�S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$:d:d:d:d:d:d*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
���������a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:0transpose_4/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:1transpose_5/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:�Na
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:dd[
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:�N[
	Reshape_7Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:d[
	Reshape_8Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:d[
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:d\

Reshape_10Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:d\

Reshape_11Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:d\

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:dM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    �
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*?
_output_shapes-
+:���������
d:
d: :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:
d*
shrink_axis_maske
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*+
_output_shapes
:
���������dg
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
_output_shapes

:
d*
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @U
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes

:
d]

Identity_1Identitytranspose_7:y:0*
T0*+
_output_shapes
:
���������dQ

Identity_2IdentitySqueeze:output:0*
T0*
_output_shapes

:
dI

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "#
concat_axisconcat/axis:output:0"
concatconcat_0:output:0"!

cudnnrnn_0CudnnRNN:output_c:0"&

cudnnrnn_1CudnnRNN:reserve_space:0"!

cudnnrnn_2CudnnRNN:output_h:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"
cudnnrnnCudnnRNN:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:
���������d:
d:	d�:	d�:	�*<
api_implements*(gru_147464ad-d9c1-4bc4-af9c-aa73203e3343*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_2376079_2376214*
go_backwards( *

time_major( :EA

_output_shapes
:	�

_user_specified_namebias:QM

_output_shapes
:	d�
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	d�
 
_user_specified_namekernel:FB

_output_shapes

:
d
 
_user_specified_nameinit_h:S O
+
_output_shapes
:
���������d
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
K
embedding_input8
!serving_default_embedding_input:0
���������=
dense4
StatefulPartitionedCall:0
���������Ltensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
J
0
%1
&2
'3
#4
$5"
trackable_list_wrapper
J
0
%1
&2
'3
#4
$5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
�
-trace_0
.trace_12�
,__inference_sequential_layer_call_fn_2375327
,__inference_sequential_layer_call_fn_2375346�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z-trace_0z.trace_1
�
/trace_0
0trace_12�
G__inference_sequential_layer_call_and_return_conditional_losses_2374918
G__inference_sequential_layer_call_and_return_conditional_losses_2375308�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z/trace_0z0trace_1
�B�
"__inference__wrapped_model_2373725embedding_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
1
_variables
2_iterations
3_learning_rate
4_index_dict
5
_momentums
6_velocities
7_update_step_xla"
experimentalOptimizer
,
8serving_default"
signature_map
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
>trace_02�
+__inference_embedding_layer_call_fn_2375417�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z>trace_0
�
?trace_02�
F__inference_embedding_layer_call_and_return_conditional_losses_2375426�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z?trace_0
&:$Ld2embedding/embeddings
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

@states
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ftrace_0
Gtrace_1
Htrace_2
Itrace_32�
%__inference_gru_layer_call_fn_2375439
%__inference_gru_layer_call_fn_2375452
%__inference_gru_layer_call_fn_2375465
%__inference_gru_layer_call_fn_2375478�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zFtrace_0zGtrace_1zHtrace_2zItrace_3
�
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_32�
@__inference_gru_layer_call_and_return_conditional_losses_2375847
@__inference_gru_layer_call_and_return_conditional_losses_2376216
@__inference_gru_layer_call_and_return_conditional_losses_2376585
@__inference_gru_layer_call_and_return_conditional_losses_2376954�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zJtrace_0zKtrace_1zLtrace_2zMtrace_3
"
_generic_user_object
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T_random_generator

%kernel
&recurrent_kernel
'bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
Ztrace_02�
'__inference_dense_layer_call_fn_2376963�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zZtrace_0
�
[trace_02�
B__inference_dense_layer_call_and_return_conditional_losses_2376993�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z[trace_0
:dL2dense/kernel
:L2
dense/bias
&:$	d�2gru/gru_cell/kernel
0:.	d�2gru/gru_cell/recurrent_kernel
$:"	�2gru/gru_cell/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
\0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_sequential_layer_call_fn_2375327embedding_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_layer_call_fn_2375346embedding_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_layer_call_and_return_conditional_losses_2374918embedding_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_layer_call_and_return_conditional_losses_2375308embedding_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
~
20
]1
^2
_3
`4
a5
b6
c7
d8
e9
f10
g11
h12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
J
]0
_1
a2
c3
e4
g5"
trackable_list_wrapper
J
^0
`1
b2
d3
f4
h5"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
%__inference_signature_wrapper_2375410embedding_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_embedding_layer_call_fn_2375417inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_embedding_layer_call_and_return_conditional_losses_2375426inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
i0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_gru_layer_call_fn_2375439inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_gru_layer_call_fn_2375452inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_gru_layer_call_fn_2375465inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_gru_layer_call_fn_2375478inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_gru_layer_call_and_return_conditional_losses_2375847inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_gru_layer_call_and_return_conditional_losses_2376216inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_gru_layer_call_and_return_conditional_losses_2376585inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_gru_layer_call_and_return_conditional_losses_2376954inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_layer_call_fn_2376963inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_layer_call_and_return_conditional_losses_2376993inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
o	variables
p	keras_api
	qtotal
	rcount"
_tf_keras_metric
+:)Ld2Adam/m/embedding/embeddings
+:)Ld2Adam/v/embedding/embeddings
+:)	d�2Adam/m/gru/gru_cell/kernel
+:)	d�2Adam/v/gru/gru_cell/kernel
5:3	d�2$Adam/m/gru/gru_cell/recurrent_kernel
5:3	d�2$Adam/v/gru/gru_cell/recurrent_kernel
):'	�2Adam/m/gru/gru_cell/bias
):'	�2Adam/v/gru/gru_cell/bias
#:!dL2Adam/m/dense/kernel
#:!dL2Adam/v/dense/kernel
:L2Adam/m/dense/bias
:L2Adam/v/dense/bias
:
d2gru/Variable
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
q0
r1"
trackable_list_wrapper
-
o	variables"
_generic_user_object
:  (2total
:  (2count�
"__inference__wrapped_model_2373725vi%&'#$8�5
.�+
)�&
embedding_input
���������
� "1�.
,
dense#� 
dense
���������L�
B__inference_dense_layer_call_and_return_conditional_losses_2376993k#$3�0
)�&
$�!
inputs
���������d
� "0�-
&�#
tensor_0
���������L
� �
'__inference_dense_layer_call_fn_2376963`#$3�0
)�&
$�!
inputs
���������d
� "%�"
unknown
���������L�
F__inference_embedding_layer_call_and_return_conditional_losses_2375426f/�,
%�"
 �
inputs
���������
� "0�-
&�#
tensor_0
���������d
� �
+__inference_embedding_layer_call_fn_2375417[/�,
%�"
 �
inputs
���������
� "%�"
unknown
���������d�
@__inference_gru_layer_call_and_return_conditional_losses_2375847�i%&'F�C
<�9
+�(
&�#
inputs_0
���������d

 
p

 
� "0�-
&�#
tensor_0
���������d
� �
@__inference_gru_layer_call_and_return_conditional_losses_2376216�i%&'F�C
<�9
+�(
&�#
inputs_0
���������d

 
p 

 
� "0�-
&�#
tensor_0
���������d
� �
@__inference_gru_layer_call_and_return_conditional_losses_2376585yi%&'?�<
5�2
$�!
inputs
���������d

 
p

 
� "0�-
&�#
tensor_0
���������d
� �
@__inference_gru_layer_call_and_return_conditional_losses_2376954yi%&'?�<
5�2
$�!
inputs
���������d

 
p 

 
� "0�-
&�#
tensor_0
���������d
� �
%__inference_gru_layer_call_fn_2375439ui%&'F�C
<�9
+�(
&�#
inputs_0
���������d

 
p

 
� "%�"
unknown
���������d�
%__inference_gru_layer_call_fn_2375452ui%&'F�C
<�9
+�(
&�#
inputs_0
���������d

 
p 

 
� "%�"
unknown
���������d�
%__inference_gru_layer_call_fn_2375465ni%&'?�<
5�2
$�!
inputs
���������d

 
p

 
� "%�"
unknown
���������d�
%__inference_gru_layer_call_fn_2375478ni%&'?�<
5�2
$�!
inputs
���������d

 
p 

 
� "%�"
unknown
���������d�
G__inference_sequential_layer_call_and_return_conditional_losses_2374918}i%&'#$@�=
6�3
)�&
embedding_input
���������
p

 
� "0�-
&�#
tensor_0
���������L
� �
G__inference_sequential_layer_call_and_return_conditional_losses_2375308}i%&'#$@�=
6�3
)�&
embedding_input
���������
p 

 
� "0�-
&�#
tensor_0
���������L
� �
,__inference_sequential_layer_call_fn_2375327ri%&'#$@�=
6�3
)�&
embedding_input
���������
p

 
� "%�"
unknown
���������L�
,__inference_sequential_layer_call_fn_2375346ri%&'#$@�=
6�3
)�&
embedding_input
���������
p 

 
� "%�"
unknown
���������L�
%__inference_signature_wrapper_2375410�i%&'#$K�H
� 
A�>
<
embedding_input)�&
embedding_input
���������"1�.
,
dense#� 
dense
���������L