# iris_pipeline.py
# ============================================================================
# SageMaker Pipeline with CodePipeline, Step Functions, and Model Monitoring
# ============================================================================

import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.model import Model
from sagemaker.model_monitor import DataCaptureConfig, ModelMonitor, Baseline
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.check_step import QualityCheckStep
from sagemaker.workflow.step_collections import ModelStep

# Setup
region = boto3.Session().region_name
role = sagemaker.get_execution_role()
sagemaker_session = sagemaker.session.Session()
pipeline_session = PipelineSession()
bucket = sagemaker_session.default_bucket()

# Parameters
training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.large")
model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.9)

# Paths
train_data_uri = f"s3://{bucket}/iris-spot-training/"

# Estimator
sklearn_estimator = SKLearn(
    entry_point="iris_sklearn_train.py",
    role=role,
    instance_type=training_instance_type,
    framework_version="0.23-1",
    py_version="py3",
    use_spot_instances=True,
    max_run=3600,
    max_wait=7200,
    hyperparameters={"n-estimators": 100, "max-depth": 5},
    sagemaker_session=pipeline_session
)

# Training Step
train_step = TrainingStep(
    name="TrainIrisModel",
    estimator=sklearn_estimator,
    inputs={"training": sagemaker.inputs.TrainingInput(train_data_uri, content_type="text/csv")}
)

# Evaluation Step
script_processor = ScriptProcessor(
    image_uri=sklearn_estimator.training_image_uri(),
    command=["python3"],
    role=role,
    instance_type="ml.m5.large",
    sagemaker_session=pipeline_session
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

eval_step = ProcessingStep(
    name="EvaluateModel",
    processor=script_processor,
    inputs=[
        ProcessingInput(source=train_data_uri + "iris_test.csv", destination="/opt/ml/processing/test"),
        ProcessingInput(source=train_step.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
    ],
    code="evaluate.py",
    property_files=[evaluation_report]
)

# Register Step
register_step = RegisterModel(
    name="RegisterTrainedModel",
    estimator=sklearn_estimator,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    approval_status=model_approval_status,
    model_package_group_name="IrisModelGroup",
    model_metrics={
        "EvaluationMetrics": {
            "EvaluationReport": {
                "S3Uri": eval_step.outputs[0].destination + "/evaluation.json"
            }
        }
    }
)

# Conditional Step
cond_step = ConditionStep(
    name="CheckModelAccuracy",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=eval_step.properties.PropertyFiles["EvaluationReport"].get("accuracy"),
            right=accuracy_threshold
        )
    ],
    if_steps=[register_step],
    else_steps=[]
)

# Model Monitor Baseline Step
monitor = ModelMonitor(role=role, instance_count=1, instance_type="ml.m5.large")
check_config = CheckJobConfig(role=role, instance_type="ml.m5.large")

data_quality_step = QualityCheckStep(
    name="DataQualityCheck",
    quality_check_config=monitor.latest_baselining_job_config(),
    check_job_config=check_config,
    skip_check=False,
    register_new_baseline=True,
    baseline_used_for_monitoring=True,
    supplied_baseline_statistics=None,
    supplied_baseline_constraints=None
)

# Model Step (for drift detection deployment)
model = Model(
    image_uri=sklearn_estimator.training_image_uri(),
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=pipeline_session,
    role=role,
    entry_point="iris_sklearn_train.py"
)

model_step = ModelStep(
    name="DeployModelWithMonitoring",
    step_args=model.create(instance_type="ml.t2.medium"),
    depends_on=[register_step.name]
)

# Pipeline Definition
pipeline = Pipeline(
    name="IrisModelPipeline",
    parameters=[training_instance_type, model_approval_status, accuracy_threshold],
    steps=[train_step, eval_step, cond_step, data_quality_step, model_step],
    sagemaker_session=pipeline_session
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    execution.wait()
    print("Pipeline execution complete.")

