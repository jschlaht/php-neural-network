<?php

namespace App\Service;

use Exception;

class NeuralNetwork
{
    private array $weightsInputToHidden;
    private array $deltaWeightsInputToHidden;

    public function getDeltaWeightsInputToHidden(): array
    {
        return $this->deltaWeightsInputToHidden;
    }

    public function getDeltaWeightsHiddenToOutput(): array
    {
        return $this->deltaWeightsHiddenToOutput;
    }
    private array $weightsHiddenToOutput;
    private array $deltaWeightsHiddenToOutput;
    private array $hiddenInputs;
    private array $hiddenOutputs;
    private array $finalInputs;
    private array $finalOutputs;
    private array $outputErrors;
    private array $hiddenErrors;

    public function __construct(
        private int $inputNodes,
        private int $hiddenNodes,
        private int $outputNodes,
        private float $learningRate
    )
    {
        $this->weightsInputToHidden = $this->createWeightsMatrixNormalDistribution($this->hiddenNodes, $this->inputNodes);
        $this->weightsHiddenToOutput = $this->createWeightsMatrixNormalDistribution($this->outputNodes, $this->hiddenNodes);
    }

    public function getInputNodes(): int
    {
        return $this->inputNodes;
    }

    public function getHiddenInputs(): array
    {
        return $this->hiddenInputs;
    }

    public function getHiddenOutputs(): array
    {
        return $this->hiddenOutputs;
    }

    public function getFinalInputs(): array
    {
        return $this->finalInputs;
    }

    public function getFinalOutputs(): array
    {
        return $this->finalOutputs;
    }

    public function getOutputErrors(): array
    {
        return $this->outputErrors;
    }

    public function getHiddenErrors(): array
    {
        return $this->hiddenErrors;
    }

    public function getHiddenNodes(): int
    {
        return $this->hiddenNodes;
    }

    public function getOutputNodes(): int
    {
        return $this->outputNodes;
    }

    public function getLearningRate(): float
    {
        return $this->learningRate;
    }

    public function getWeightsInputToHidden(): array
    {
        return $this->weightsInputToHidden;
    }

    public function getWeightsHiddenToOutput(): array
    {
        return $this->weightsHiddenToOutput;
    }

    public function setWeightsInputToHidden(array $weightsInputToHidden): void
    {
        $this->weightsInputToHidden = $weightsInputToHidden;
    }

    public function setWeightsHiddenToOutput(array $weightsHiddenToOutput): void
    {
        $this->weightsHiddenToOutput = $weightsHiddenToOutput;
    }

    /**
     * @throws Exception
     */
    public function train(array $inputList, array $targetList): void
    {
        $targetValues = $this->transposeVector($targetList);

        $this->query($inputList);
        $this->outputErrors = $this->calculateDiff($targetValues, $this->finalOutputs);
        $this->hiddenErrors = $this->dotProduct($this->transposeMatrix($this->weightsHiddenToOutput), $this->outputErrors);

        $result1 = $this->elementWiseMatrixOperation($this->outputErrors, $this->finalOutputs);
        $result2 = $this->scalarToMatrixOperation(1, $this->finalOutputs, 'subtract');
        $result3 = $this->elementWiseMatrixOperation($result1, $result2);
        $result4 = $this->dotProduct($result3, $this->transposeMatrix($this->hiddenOutputs));
        $this->deltaWeightsHiddenToOutput = $this->scalarToMatrixOperation($this->learningRate, $result4);
        $this->weightsHiddenToOutput = $this->elementWiseMatrixOperation($this->weightsHiddenToOutput, $this->deltaWeightsHiddenToOutput, 'add');

        $result5 = $this->elementWiseMatrixOperation($this->hiddenErrors, $this->hiddenOutputs);
        $result6 = $this->scalarToMatrixOperation(1, $this->hiddenOutputs, 'subtract');
        $result7 = $this->elementWiseMatrixOperation($result5, $result6);
        $result8 = $this->dotProduct($result7, $inputList);
        $this->deltaWeightsInputToHidden = $this->scalarToMatrixOperation($this->learningRate, $result8);
        $this->weightsInputToHidden = $this->elementWiseMatrixOperation($this->weightsInputToHidden, $this->deltaWeightsInputToHidden, 'add');
    }

    /**
     * @throws Exception
     */
    public function query(array $inputList): void
    {
        $inputValues = $this->transposeVector($inputList);
        $this->hiddenInputs = $this->dotProduct($this->weightsInputToHidden, $inputValues);
        $this->hiddenOutputs = $this->doActivate($this->hiddenInputs);

        $this->finalInputs = $this->dotProduct($this->weightsHiddenToOutput, $this->hiddenOutputs);
        $this->finalOutputs = $this->doActivate($this->finalInputs);
    }

    public function dotProduct(array $matrix1, array $matrix2): array
    {
        if (count($matrix1[0]) != count($matrix2)) {
            throw new \Exception("Matrix dimensions do not match");
        }
        $result = array();
        for ($i = 0; $i < count($matrix1); $i++) {
            $result[$i] = array();
            for ($j = 0; $j < count($matrix2[0]); $j++) {
                $sum = 0;
                for ($k = 0; $k < count($matrix2); $k++) {
                    $sum += $matrix1[$i][$k] * $matrix2[$k][$j];
                }
                $result[$i][$j] = round($sum,3);
            }
        }
        return $result;
    }

    public function doActivate($vector): array
    {
        return $this->calculateActivationValue($this->transposeVector($vector));
    }
    public function doCalculateDifference($vector1, $vector2): array
    {
        return $this->calculateDiff($vector1, $vector2);
    }

    private function createWeightsMatrixNormalDistribution(int $rows, int $cols, $mean = 0, $standard_deviation = 0.5): array
    {
        $matrix = [];
        for ($i = 0; $i < $rows; $i++) {
            for ($j = 0; $j < $cols; $j++) {
                $u1 = mt_rand() / mt_getrandmax();
                $u2 = mt_rand() / mt_getrandmax();
                $z = sqrt(-2 * log($u1)) * cos(2 * pi() * $u2);
                $normalized_z = ($z * $standard_deviation) + $mean;
                $clamped_value = max(-0.99, min(0.99, $normalized_z));
                $matrix[$i][$j] = $clamped_value;
            }
        }
        return $matrix;
    }

    private function calculateActivationValue($vector): array
    {
        $result = array();
        for ($i = 0; $i < count($vector); $i++) {
            $result[$i][0] = round(1 / (1 + exp(-$vector[$i][0])), 3);
        }
        return $result;
    }

    private function transposeVector($vector)
    {
        if (count($vector) > 1 && is_numeric($vector[0])) {
            $transposedVector = array_map(fn($x) => [$x], $vector);
        } else {
            $transposedVector = $vector;
        }
        return $transposedVector;
    }

    private function transposeMatrix($matrix): array
    {
        return array_map(null, ...$matrix);
    }

    private function calculateDiff(array $target, array $final): array
    {
        $result = array_map(
            fn($x, $y) => round($x[0] - $y[0], 3), $target, $final);
        return $this->transposeVector($result);
    }

    private function scalarToMatrixOperation(float $scalar, array $matrix, string $operation = 'multiply'): array
    {
        $result = [];

        foreach ($matrix as $row) {
            $newRow = [];
            foreach ($row as $element) {
                $newRow[] = match ($operation) {
                    'divide' => $scalar / $element,
                    'subtract' => $scalar - $element,
                    'add' => $scalar + $element,
                    default => $scalar * $element,
                };
            }
            $result[] = $newRow;
        }

        return $result;
    }

    private function elementWiseMatrixOperation(array $matrix1, array $matrix2, string $operation = 'multiply'): array
    {
        if (count($matrix1) !== count($matrix2) || count($matrix1[0]) !== count($matrix2[0])) {
            throw new Exception("Both matrices must have the same dimensions.");
        }

        $result = [];

        for ($i = 0; $i < count($matrix1); $i++) {
            $newRow = [];
            for ($j = 0; $j < count($matrix1[0]); $j++) {
                $newRow[] = match ($operation) {
                    'divide' => $matrix1[$i][$j] / $matrix2[$i][$j],
                    'subtract' => $matrix1[$i][$j] - $matrix2[$i][$j],
                    'add' => $matrix1[$i][$j] + $matrix2[$i][$j],
                    default => $matrix1[$i][$j] * $matrix2[$i][$j]
                };
            }
            $result[] = $newRow;
        }

        return $result;
    }
}
