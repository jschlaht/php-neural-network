<?php

namespace App\Service;

class NeuralNetwork
{
    private array $weightsInputToHidden;
    private array $weightsHiddenToOutput;

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

    public function train()
    {

    }

    public function query(array $inputList): array
    {
        $inputValues = $this->transposeVector($inputList);
        $hiddenInputs = $this->dotProduct($this->weightsInputToHidden, $inputValues);
        $hiddenOutputs = $this->doActivate($hiddenInputs);

        $finalInputs = $this->dotProduct($this->weightsHiddenToOutput, $hiddenOutputs);
        $finalOutputs = $this->doActivate($finalInputs);

        return $finalOutputs;
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
                $result[$i][$j] = $sum;
            }
        }
        return $result;
    }

    public function doActivate($vector): array
    {
        return $this->calculateActivationValue($this->transposeVector($vector));
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
}
